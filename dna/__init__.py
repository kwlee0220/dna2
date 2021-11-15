import os
from typing import Union
from pathlib import Path

from omegaconf import OmegaConf

from .types import Point, Size2d, Box
from .utils import get_logger

DNA_HOME = Path(os.environ.get('DNA_HOME', '.'))
DNA_CONIFIG_FILE = DNA_HOME / 'conf' / 'config.yaml'

DEBUG_FRAME_IDX = -1
DEBUG_SHOW_IMAGE = False
DEBUG_PRINT_COST = DEBUG_SHOW_IMAGE
DEBUG_START_FRAME = 32
DEBUG_TARGET_TRACKS = None


def load_config(base_config: Path, conf_id: str =None) -> OmegaConf:
    conf = OmegaConf.load(base_config)
    if conf_id:
        node_conf = load_sub_config(base_config.parent, conf_id)
        conf = OmegaConf.merge(conf, node_conf)
    return conf

def load_sub_config(root_dir: Path, conf_id: str) -> Union[OmegaConf,None]:
    suffix = conf_id.replace('.', '/') + ".yaml"
    full_path = root_dir / suffix
    if full_path.is_file():
        return OmegaConf.load(full_path)
    else:
        raise ValueError(f"configuration file is not found: '{full_path}'")

def traverse_configs(root_dir: Path):
    prefix_len = len(Path(os.path.splitext(root_dir)[0]).parts)
    for dir, subdirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".yaml":
                full_path = os.path.join(dir, file)
                parts = Path(os.path.splitext(full_path)[0]).parts[prefix_len:]
                conf_path = '.'.join(parts)
                yield (conf_path, Path(full_path))

from collections import defaultdict
def parse_config_args(args): 
    config_grp = defaultdict(list) 
    for arg in args:
        idx = arg.find('=')
        if idx >= 0:
            key = arg[:idx]
            value = arg[idx+1:]
            idx = key.find('.')
            grp_key = key[:idx]
            key = key[idx+1:]
            config_grp[grp_key].append((key, value))
    return config_grp