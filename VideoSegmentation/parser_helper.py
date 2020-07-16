import yaml,os
from bunch import Bunch,bunchify
import argparse

def _merge_configs(config,custom_config):
    for key in custom_config.keys():
        config[key] = custom_config[key]
    return config


def parse_args (parser):
    #read default config
    args=parser.parse_args()
    # load model config
    default_path = args.default_path
    with open(default_path,"rb") as f:
        config = yaml.safe_load(f)
    #read current config
    custom_path = args.custom_path
    with open(custom_path,"rb") as f:
        custom_config = yaml.safe_load(f)
    #merge
    config = _merge_configs(config,custom_config)
    #bunchify
    opt = Bunch( (k, bunchify(v)) for k,v in config.items() )

    return opt
    