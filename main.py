import argparse
import datetime
import random
import numpy as np
import torch
import yaml
import logging
import algorithm
from handle import Handle

logger = logging.getLogger("logger")


def seed(seeds):
    np.random.seed(seeds)
    torch.manual_seed(seeds)
    torch.cuda.manual_seed(seeds)
    random.seed(seeds)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated learning attack and defence')
    parser.add_argument('--params', default='cifar_params.yaml', dest='params')
    parser.add_argument('--defense', default='None', dest='defense')
    args = parser.parse_args()
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader)
    seed(params_loaded['seed'])
    logger.info(f'Start training')
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    params_loaded['defence_method'] = args.defense
    handle = Handle(current_time=current_time, params=params_loaded,
                    name=params_loaded.get('name', format(params_loaded['type'])))
    handle.load_data()
    logger.info(f'load data done')
    handle.create_model()
    logger.info(handle.model)
    logger.info(f'create model done')
    if handle.params['is_poison']:
        logger.info(f"Poisoned following participants: {handle.adversarial_namelist}")
    with open(f'{handle.folder_path}/params.yaml', 'w') as f:
        yaml.dump(handle.params, f)
    if handle.params['algorithm'] == 'FedAvg':
        algorithm.FedAvg(handle)
