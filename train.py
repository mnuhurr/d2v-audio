

from pathlib import Path
from copy import deepcopy

import math
import time
import torch
import torch.nn.functional as F

import audiofile as af

from common import init_log, read_yaml
from dataset import RawAudioDataset
from dataset import MelDataset
from models import D2VEncoder
from models.utils import model_size, ema_update
from models.masking import simulate_masking

from typing import List, Union, Optional, Any


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_dir_filenames(dir_path: Union[str, Path], ext: str = '.wav', min_duration: Optional[float] = None) -> List[Path]:
    filenames = list(Path(dir_path).glob(f'*{ext}'))

    if min_duration is not None:
        filenames = list(filter(lambda fn: af.duration(fn) >= min_duration, filenames))

    return filenames


def mask_loss(y_pred, y_true, mask, loss_object=F.mse_loss):
    loss = loss_object(y_pred, y_true, reduction='none')
    # count loss from the masked positions
    mask = (mask < 0).to(loss.dtype)
    mask = mask.unsqueeze(-1)
    return torch.sum(loss * mask) / torch.sum(mask).to(loss.dtype)


def step_lr(step, d_model, warmup_steps=4000):
    # learning rate from the original attention paper
    arg1 = torch.tensor(1 / math.sqrt(step)) if step > 0 else torch.tensor(float('inf'))
    arg2 = torch.tensor(step * warmup_steps**-1.5)

    return 1 / math.sqrt(d_model) * torch.minimum(arg1, arg2)


def avg_var(x: torch.Tensor) -> torch.Tensor:
    vars = torch.var(x, dim=-2)
    return torch.mean(vars)


def normalize_block(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    s = x.std(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    return (x - mu) / s


def loss_fn(y_student: List[torch.Tensor], y_teacher: List[torch.Tensor], mask: torch.Tensor, n_layers: int = 8, lambda_var: float = 1.0):
    #avg_student = torch.mean(torch.stack(y_student[-n_layers:]), dim=0)
    avg_teacher = torch.mean(torch.stack(y_teacher[-n_layers:]), dim=0)

    var_loss = F.relu(1 - avg_var(y_teacher[-1]))

    loss = mask_loss(y_student[-1], avg_teacher, mask) + lambda_var * var_loss

    return loss


def train(model: torch.nn.Module,
          target: torch.nn.Module,
          loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: Optional[Any] = None,
          log_interval: Optional[int] = 100,
          n_layers: int = 8,
          ema_decay: float = 0.999,
          lambda_var: float = 1.0):

    model.train()
    target.train()

    train_loss = 0.0

    batch_t0 = time.time()

    for batch, x in enumerate(loader):
        x = x.to(device)

        y_student, mask = model(x, masking=True)
        with torch.no_grad():
            y_teacher, _ = target(x, masking=False)

        # take the n_layers last layers for comparison
        #losses = []
        #for u, v in zip(y_student[-n_layers:], y_teacher[-n_layers:]):
        #    losses.append(mask_loss(u, v, mask))
        #loss = sum(losses)

        # normalize teacher
        y_teacher = [normalize_block(block) for block in y_teacher]

        loss = loss_fn(y_student, y_teacher, mask, n_layers=n_layers, lambda_var=lambda_var)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # ema update
        ema_update(model, target, ema_decay)

        if log_interval is not None and batch % log_interval == 0:
            t_batch = (time.time() - batch_t0) * 1000 / log_interval
            current_lr = optimizer.param_groups[0]['lr']
            print(f'batch {batch:5d}/{len(loader)} | {int(t_batch):4d} ms/batch | learning rate {current_lr:.4g} | train loss {loss.item():.4f} | avg coordinate var {avg_var(y_teacher[-1]):.4f}')
            batch_t0 = time.time()

    return train_loss / len(loader)


@torch.no_grad()
def validate(model: torch.nn.Module,
             target: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             n_layers: int = 8,
             lambda_var: float = 1.0):

    model.eval()
    target.eval()

    val_loss = 0.0
    val_vars = 0.0

    for x in loader:
        x = x.to(device)

        y_student, mask = model(x, masking=True)
        y_teacher, _ = target(x, masking=False)

        # take the n_layers last layers for comparison
        #losses = []
        #for u, v in zip(y_student[-n_layers:], y_teacher[-n_layers:]):
        #    losses.append(mask_loss(u, v, mask))
        #loss = sum(losses)
        
        # normalize teacher
        y_teacher = [normalize_block(block) for block in y_teacher]

        loss = loss_fn(y_student, y_teacher, mask, n_layers=n_layers, lambda_var=lambda_var)

        val_loss += loss.item()
        val_vars += avg_var(y_teacher[-1])

    return val_loss / len(loader), val_vars / len(loader)


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('trainer', filename='train.log', level=cfg.get('log_level', 'info'))

    min_duration = cfg.get('min_duration')
    max_duration = cfg.get('max_duration', 10.0)
    sample_rate = cfg.get('sample_rate', 16000)

    batch_size = cfg.get('batch_size', 16)
    num_workers = cfg.get('num_dataloader_workers', 2)
    epochs = cfg.get('epochs', 10)

    log_interval = cfg.get('log_interval', 100)
    max_patience = cfg.get('patience', 5)

    model_path = Path(cfg.get('model_path', 'model.pt'))
    model_path.parent.mkdir(exist_ok=True, parents=True)

    train_files = []
    val_files = []

    logger.info('reading files')
    for train_dir in cfg.get('train_data_dirs', []):
        train_files.extend(get_dir_filenames(train_dir, min_duration=min_duration))

    for val_dir in cfg.get('val_data_dirs', []):
        val_files.extend(get_dir_filenames(val_dir, min_duration=min_duration))

    logger.info(f'got {len(train_files)} for training, {len(val_files)} for validation')

    n_fft = cfg.get('n_fft', 1024)
    hop_length = cfg.get('hop_length')
    n_mels = cfg.get('n_mels', 64)
    max_length = 250

    #ds_train = RawAudioDataset(train_files, sample_rate=sample_rate, max_duration=max_duration)
    #ds_val = RawAudioDataset(val_files, sample_rate=sample_rate, max_duration=max_duration)
    ds_train = MelDataset(
        filenames=train_files, 
        sample_rate=sample_rate, 
        max_length=max_length, 
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length)

    ds_val = MelDataset(
        filenames=val_files, 
        sample_rate=sample_rate, 
        max_length=max_length, 
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length)

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers)
    
    # start building models
    d_model = cfg.get('d_model')
    d_ff = cfg.get('d_ff')
    n_heads = cfg.get('n_heads')
    n_layers = cfg.get('n_layers')
    
    max_sequence_length = cfg.get('max_sequence_length', 256)

    p_masking = cfg.get('p_masking', 0.065)
    masking_length = cfg.get('masking_length', 10)
    p_token_mask = simulate_masking(p_masking, masking_length, num_timesteps=16)
    logger.info(f'p_masking={p_masking}, masking_length={masking_length}, fraction of masked tokens approx {p_token_mask:.3f}')

    learning_rate = cfg.get('learning_rate_factor', 1.0)
    ema_decay = cfg.get('ema_decay', 0.999)
    lambda_var = cfg.get('lambda_var', 1.0)
    warmup = cfg.get('warmup_steps', 4000)
    
    logger.info(f'learning_rate_factor={learning_rate}, ema_decay={ema_decay}, warmup_steps={warmup}, lambda_var={lambda_var}')
    logger.info(f'd_model={d_model}, d_ff={d_ff}, n_heads={n_heads}, n_layers={n_layers}')

    # encoder
    model = D2VEncoder(
        d_model=d_model,
        n_layers=n_layers,
        d_ff=d_ff,
        n_heads=n_heads,
        n_mels=n_mels,
        max_sequence_length=max_sequence_length,
        p_masking=p_masking,
        masking_length=masking_length)
    target = deepcopy(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step_lr(step, d_model, warmup_steps=warmup))
    logger.info(f'model size {model_size(model)/1e6:.1f}M')

    model = model.to(device)
    target = target.to(device)

    patience = max_patience
    best_loss = float('inf')

    for epoch in range(epochs):
        t0 = time.time()

        train_loss = train(model, target, train_loader, optimizer, scheduler, log_interval=log_interval, ema_decay=ema_decay, lambda_var=lambda_var)
        dt = time.time() - t0
        logger.info(f'epoch {epoch + 1} training done in {dt:.1f} seconds, training loss {train_loss:.4f}')

        val_loss, val_var = validate(model, target, val_loader, lambda_var=lambda_var)
        logger.info(f'epoch {epoch + 1} validation loss {val_loss:.4f}, avg variance {val_var:.4f}')

        if val_loss < best_loss:
            patience = max_patience
            best_loss = val_loss
            # save
            torch.save(model.state_dict(), model_path)

        else:
            patience -= 1
            if patience <= 0:
                logger.info(f'results not improving, stopping')
                break

if __name__ == '__main__':
    main()
