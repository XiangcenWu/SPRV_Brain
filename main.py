import argparse
from utils.config import MainConfig
import os
import logging



def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # Common params
    parser.add_argument("-c", "--config", help="The path of config file.", type=str)

    return parser.parse_args()

def config_log(cfg):
    FORMAT = '%(asctime)s, %(message)s'
    logging.basicConfig(
        filename=os.path.join(cfg.result_dir, cfg.nick_name+'.txt'),
        level=logging.INFO,
        filemode='w',
        format=FORMAT
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def make_result_dir(cfg):
    cfg.add_dict_item({'result_dir': os.path.join(cfg.result_base_dir, cfg.nick_name)})
    if not os.path.exists(cfg.result_base_dir):
        os.mkdir(cfg.result_base_dir)
    if not os.path.exists(cfg.result_dir):
        os.mkdir(cfg.result_dir)
        
    


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = MainConfig(args.config)
    make_result_dir(cfg)
    
    
    config_log(cfg)
    logging.info('test')
    logging.debug('fdsafsdgsgdsa')
    logging.warning('fdsafsd')



if __name__ == '__main__':
    args = parse_args()
    main(args)