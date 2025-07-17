import os, yaml

def get_cfg(file):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, "..", file)
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config
