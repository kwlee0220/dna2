from typing import Tuple
from pathlib import Path

def _parse_keyvalue(kv) -> Tuple[str,str]:
    pair = kv.split('=')
    if len(pair) == 2:
        return tuple(pair)
    else:
        return pair, None

def parse_query(query):
    if not query or len(query) == 0:
        return dict()
    return dict([_parse_keyvalue(kv) for kv in query.split(':')])

def get_first_param(args, key, def_value=None):
    value = args.get(key)
    return value[0] if value else def_value

def get_first_param_path(args, key, def_value=None):
    value = args.get(key)
    if value:
        return Path(value[0])
    elif def_value:
        return Path(def_value)
    else:
        return None


import logging
_LOGGERS = dict()
_LOG_FORMATTER = logging.Formatter("%(levelname)s: %(message)s (%(filename)s)")

def get_logger(name=None):
    logger = _LOGGERS.get(name)
    if not logger:
        logger = logging.getLogger(name)
        _LOGGERS[name] = logger
        
        logger.setLevel(logging.DEBUG)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(_LOG_FORMATTER)
        logger.addHandler(console)
        
    return logger