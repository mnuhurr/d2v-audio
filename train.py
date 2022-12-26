

from pathlib import Path
from copy import deepcopy

import math
import time
import torch
import torch.nn.functional as F

import audiofile as af
import pandas as pd

from common import init_log, read_yaml
from dataset import RawAudioDataset, MelDataset, collate_fn
from models import D2VEncoder
from models.utils import model_size, ema_update
from models.masking import simulate_masking

from typing import List, Union, Optional, Any, Tuple


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_dir_filenames(dir_path: Union[str, Path], ext: str = '.wav', min_duration: Optional[float] = None) -> List[Path]:
    """
    get list of suitable files in a directory. the file durations are not read if min_duration is not given (improves
    speed significantly)

    :param dir_path: directory containing the files
    :param ext: file extension ('.wav' by default)
    :param min_duration: (optional) minimum duration (in seconds)
    :return: list of Path objects
    """
    filenames = list(Path(dir_path).glob(f'*{ext}'))

    if min_duration is not None:
        filenames = list(filter(lambda fn: af.duration(fn) >= min_duration, filenames))

    return filenames


def smoothed_l1_loss(y_pred: torch.Tensor, y_true: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    smoothed loss from the paper. for values with absolute error at most beta returns l2 loss, otherwise l1.
    this function does not do any reduction.
    
    :param y_pred:
    :param y_true:
    :param beta: threshold value for smoothing
    :return: tensor of smoothed losses
    """
    # no reduction
    e = torch.abs(y_pred - y_true)

    ind = e <= beta

    e[ind] = 0.5 * e[ind]**2 / beta
    e[~ind] = e[~ind] - 0.5 * beta

    return e


def mask_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    compute mse loss for the masked positions. masked positions are marked with negative values.

    :param y_pred:
    :param y_true:
    :param mask:
    :return:
    """
    #loss = F.mse_loss(y_pred, y_true, reduction='none')
    loss = smoothed_l1_loss(y_pred, y_true, beta=1.5)
    # count loss from the masked positions
    mask = (mask < 0).to(loss.dtype)
    mask = mask.unsqueeze(-1)
    return torch.sum(loss * mask) / torch.sum(mask).to(loss.dtype)


def loss_fn(y_student: List[torch.Tensor], y_teacher: List[torch.Tensor], mask: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, n_teacher_layers: int = 8, lambda_var: float = 1.0):
    """
    general loss function used for train/validation
    :param y_student: outputs from the student network
    :param y_teacher: outputs from the teacher network
    :param mask: mask (from the student network)
    :param mask: token padding mask (possibly contracted in the network)
    :param n_teacher_layers: number of layers to average from the teacher output
    :param lambda_var:
    :return:
    """

    # we want to take those positions that are masked out for student (mask < 0) and those that are included in the
    # input (padding_mask == 0). flip the padding mask and combine with the training mask.
    if padding_mask is not None:
        # combine mask: start from training mask, and remove those positions that are not included in the original input
        mask[padding_mask < 0] = 0
        
    if torch.sum(mask < 0) == 0:
        # nothing to take the loss from
        return None

    avg_teacher = torch.mean(torch.stack(y_teacher[-n_teacher_layers:]), dim=0)

    var_loss = F.relu(1 - avg_var(y_teacher[-1]))

    loss = mask_loss(y_student[-1], avg_teacher, mask) + lambda_var * var_loss

    return loss


def step_lr(step: int, d_model: int, warmup_steps: int = 4000) -> float:
    """
    step size from the original attention paper
    :param step:
    :param d_model:
    :param warmup_steps:
    :return:
    """
    arg1 = torch.tensor(1 / math.sqrt(step)) if step > 0 else torch.tensor(float('inf'))
    arg2 = torch.tensor(step * warmup_steps**-1.5)

    return 1 / math.sqrt(d_model) * torch.minimum(arg1, arg2)


def piecewise_linear_lr(step: int, peak_rate: float, rise_steps: int = 2000, hold_steps: int = 30000, release_steps: int = 30000):
    if step <= rise_steps:
        return step / rise_steps * peak_rate
    
    elif step <= rise_steps + hold_steps:
        return peak_rate

    else:
        f = min((step - hold_steps - rise_steps) / release_steps, 1.0)
        return (1 - f) * peak_rate
        

def avg_var(x: torch.Tensor) -> torch.Tensor:
    """
    compute average variance for the coordinate representations (i.e. dim=-2)
    :param x: batch tensor
    :return: average variance
    """
    vars = torch.var(x, dim=-2)
    return torch.mean(vars)


def normalize_block(x: torch.Tensor) -> torch.Tensor:
    """
    normalize all inputs in the batch by their mean/stddev
    :param x: batch tensor
    :return: normalized batch
    """
    #mu = x.mean(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    #s = x.std(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    #return (x - mu) / s
    return F.layer_norm(x, (x.size(-1),))


def train(model: torch.nn.Module,
          target: torch.nn.Module,
          scaler: torch.cuda.amp.GradScaler,
          loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: Optional[Any] = None,
          log_interval: Optional[int] = 100,
          n_teacher_layers: int = 8,
          ema_decay: float = 0.999,
          lambda_var: float = 0.0) -> float:
    """
    training for one epoch.

    :param model: student network
    :param target: teacher network
    :param loader: dataloader
    :param optimizer: optimizer
    :param scheduler: optional learning rate scheduler
    :param log_interval: optional interval of batches for intermediate printouts
    :param n_teacher_layers: number of layers to include in the teacher network output average
    :param ema_decay: ema decay value
    :param lambda_var: multiplier for variance loss
    :return: training loss
    """
    model.train()
    target.train()

    train_loss = 0.0

    batch_t0 = time.time()

    for batch, (x, input_mask) in enumerate(loader):
        x = x.to(device)
        input_mask = input_mask.to(device)

        with torch.cuda.amp.autocast():
            y_student, mask, padding_mask = model(x, input_mask=input_mask, mode='student')
            with torch.no_grad():
                y_teacher, _ = target(x, input_mask=input_mask, mode='teacher')

            # normalize teacher
            y_teacher = [normalize_block(block) for block in y_teacher]

            loss = loss_fn(y_student, y_teacher, mask, padding_mask=padding_mask, n_teacher_layers=n_teacher_layers, lambda_var=lambda_var)
        if loss is None:
            continue

        train_loss += loss.item()

        optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        if scheduler is not None:
            scheduler.step()
        scaler.update()

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
             n_teacher_layers: int = 8,
             lambda_var: float = 1.0) -> Tuple[float, float]:
    """
    validation for one epoch.

    :param model: student network
    :param target: teacher network
    :param loader: dataloader
    :param n_teacher_layers: number of layers to include in the teacher network output average
    :param lambda_var: multiplier for variance loss
    :return: tuple containing validation loss and avg coordinate variance
    """
    model.eval()
    target.eval()

    val_loss = 0.0
    val_vars = 0.0

    for x, input_mask in loader:
        x = x.to(device)
        input_mask = input_mask.to(device)

        y_student, mask, padding_mask = model(x, input_mask=input_mask, mode='student')
        y_teacher, _ = target(x, input_mask=input_mask, mode='teacher')

        # normalize teacher
        y_teacher = [normalize_block(block) for block in y_teacher]

        loss = loss_fn(y_student, y_teacher, mask, padding_mask=padding_mask, n_teacher_layers=n_teacher_layers, lambda_var=lambda_var)
        if loss is None:
            continue

        val_loss += loss.item()
        val_vars += avg_var(y_teacher[-1]).item()

    return val_loss / len(loader), val_vars / len(loader)


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('trainer', filename=cfg.get('train_log_filename'), level=cfg.get('log_level', 'info'))

    min_duration = cfg.get('min_duration')
    max_duration = cfg.get('max_duration', 10.0)
    sample_rate = cfg.get('sample_rate', 16000)

    batch_size = cfg.get('batch_size', 16)
    num_workers = cfg.get('num_dataloader_workers', 2)
    epochs = cfg.get('epochs', 10)

    log_interval = cfg.get('log_interval', 100)
    max_patience = cfg.get('patience')
    warmup_epochs = cfg.get('warmup_epochs')

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

    # with 16kHz sampling rate, n_fft=1024, hop_length=512 an eight-second clip will be 251 mel frames.
    n_fft = cfg.get('n_fft', 1024)
    hop_length = cfg.get('hop_length')
    n_mels = cfg.get('n_mels', 64)
    max_length = None

    ds_train = MelDataset(
        filenames=train_files, 
        sample_rate=sample_rate, 
        #max_length=max_length, 
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length)

    ds_val = MelDataset(
        filenames=val_files, 
        sample_rate=sample_rate, 
        #max_length=max_length, 
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length)
    """
    ds_train = RawAudioDataset(filenames=train_files, sample_rate=sample_rate)
    ds_val = RawAudioDataset(filenames=val_files, sample_rate=sample_rate)
    """

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    
    # start building models
    d_model = cfg.get('d_model')
    d_ff = cfg.get('d_ff')
    n_heads = cfg.get('n_heads')
    n_layers = cfg.get('n_layers')

    # this affects to the positional encoding size
    max_sequence_length = cfg.get('max_sequence_length', 256)

    p_masking = cfg.get('p_masking', 0.065)
    masking_length = cfg.get('masking_length', 10)
    p_token_mask = simulate_masking(p_masking, masking_length, num_timesteps=80)
    logger.info(f'p_masking={p_masking}, masking_length={masking_length}, fraction of masked tokens approx {p_token_mask:.3f}')

    # number of layers from the teacher network to average over
    n_teacher_layers = cfg.get('n_teacher_layers', 8)

    #learning_rate = cfg.get('learning_rate_factor', 1.0)
    learning_rate = cfg.get('learning_rate', 1e-4)
    ema_decay = cfg.get('ema_decay', 0.999)
    lambda_var = cfg.get('lambda_var', 1.0)
    warmup = cfg.get('warmup_steps', 4000)
    
    logger.info(f'learning_rate_factor={learning_rate}, ema_decay={ema_decay}, warmup_steps={warmup}, lambda_var={lambda_var}')
    logger.info(f'training targets are averages of the last {n_teacher_layers} teacher model output layers')
    logger.info(f'd_model={d_model}, d_ff={d_ff}, n_heads={n_heads}, n_layers={n_layers}')

    # student encoder
    model = D2VEncoder(
        d_model=d_model,
        n_layers=n_layers,
        d_ff=d_ff,
        n_heads=n_heads,
        n_mels=n_mels,
        max_sequence_length=max_sequence_length,
        p_masking=p_masking,
        masking_length=masking_length)

    # teacher
    target = deepcopy(model)

    num_steps = epochs * len(train_loader)
    rise_steps = int(0.05 * num_steps)
    hold_steps = int(0.85 * num_steps)
    release_steps = int(0.1 * num_steps)
    logger.info(f'piecewise linear learning rate warmup {rise_steps}, peak rate {hold_steps}, release {release_steps} steps')

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step_lr(step, d_model, warmup_steps=warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
            lambda step: piecewise_linear_lr(step, peak_rate=learning_rate, rise_steps=rise_steps, hold_steps=hold_steps, release_steps=release_steps))
    # piecewise_linear_lr(step: int, peak_rate: float, rise_steps: int = 2000, hold_steps: int = 30000, release_steps: int = 30000):
    logger.info(f'model size {model_size(model)/1e6:.1f}M')

    model = model.to(device)
    target = target.to(device)

    patience = max_patience
    best_loss = float('inf')

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_avg_var': []
    }

    logger.info(f'start training for {epochs} epochs')
    for epoch in range(epochs):
        t0 = time.time()

        train_loss = train(model, target, scaler, train_loader, optimizer, scheduler, log_interval=log_interval, ema_decay=ema_decay, lambda_var=lambda_var)
        dt = time.time() - t0
        logger.info(f'epoch {epoch + 1} training done in {dt:.1f} seconds, training loss {train_loss:.4f}')

        val_loss, val_var = validate(model, target, val_loader, lambda_var=lambda_var)
        logger.info(f'epoch {epoch + 1} validation loss {val_loss:.4f}, avg variance {val_var:.4f}')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_avg_var'].append(val_var)

        if warmup_epochs is not None and epoch < warmup_epochs:
            continue

        if val_loss < best_loss:
            torch.save(model.state_dict(), model_path)

            if max_patience is not None:
                patience = max_patience
                best_loss = val_loss

        elif max_patience is not None:
            patience -= 1
            if patience <= 0:
                logger.info(f'results not improving, stopping')
                break

    if 'final_model_path' in cfg:
        torch.save(model.state_dict(), cfg['final_model_path'])

    csv_fn = cfg.get('train_hist_filename')
    if csv_fn is not None:
        df = pd.DataFrame(history)
        df.to_csv('history.csv')


if __name__ == '__main__':
    main()
