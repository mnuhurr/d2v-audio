
from pathlib import Path

import torch
import torch.nn.functional as F

import audiofile as af


from common import init_log, read_yaml
from dataset import RawAudioDataset
from models.encoders import WaveEncoder
from models.utils import model_size

from typing import List, Union, Optional


def get_dir_filenames(dir_path: Union[str, Path], ext: str = '.wav', min_duration: Optional[float] = None) -> List[Path]:
    filenames = list(Path(dir_path).glob(f'*{ext}'))

    if min_duration is not None:
        filenames = list(filter(lambda fn: af.duration(fn) >= min_duration, filenames))

    return filenames



def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('trainer', level=cfg.get('log_level', 'info'))

    min_duration = cfg.get('min_duration')
    max_duration = cfg.get('max_duration', 10.0)
    sample_rate = cfg.get('sample_rate', 16000)

    #logger.info(f'got {len(train_files)} for training, {len(val_files)} for validation')

if __name__ == '__main__':
    main()