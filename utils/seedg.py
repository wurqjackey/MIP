from config import cfg

def seed_gen(cfg):
    if not cfg.DATASETS.NAMES == 'sysu_mm':
        return False
    if cfg.DATASETS.MODE == 'all':
        if cfg.DATASETS.SETTING == 'one':
            return 254
        else:
            return 1221
    else:
        if cfg.DATASETS.SETTING == 'all':
            return 1884
        else:
            return 2936
        
