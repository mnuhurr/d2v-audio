
from pathlib import Path

import torch
import torch.nn.functional as F


from common import init_log, read_yaml
from dataset import RawAudioDataset
from models.encoders import WaveEncoder
from models.utils import model_size

from typing import List, Union


def get_filenames(dir_list: List[Union[str, Path]], ext: str = '.wav') -> List[Path]:
    filenames = []
    for dir_path in dir_list:
        print(dir_path)
        filenames.extend(list(Path(dir_path).glob(f'*{ext}')))

    return filenames


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('trainer', level=cfg.get('log_level', 'info'))

    train_files = get_filenames(cfg.get('train_data_dirs', []))
    val_files = get_filenames(cfg.get('val_data_dirs', []))

    print(len(train_files))
    print(len(val_files))

if __name__ == '__main__':
    main()