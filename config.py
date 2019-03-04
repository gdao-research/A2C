from collections import namedtuple

_CONFIG = {
    'iterations': 100000000,
    'eval_freq': 100000,
    'eval_episodes': 30,
    'env_id': 'BreakoutNoFrameskip-v4',
    'nb_workers': 4,
    'nb_rollout': 5
}

Params = namedtuple(typename='Params', field_names=list(_CONFIG.keys()))
CONFIG = Params(**_CONFIG)
