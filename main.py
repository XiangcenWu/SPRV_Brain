import argparse
from utils.config import MainConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # Common params
    parser.add_argument("-c", "--config", help="The path of config file.", type=str)

    return parser.parse_args()



def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = MainConfig(args.config)
    print(cfg.others.device)

if __name__ == '__main__':
    args = parse_args()
    main(args)