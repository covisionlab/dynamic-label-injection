from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Train a model')
    
    # required
    parser.add_argument('--seed', type=int, required=True, help='seed')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--method', type=str, required=True, help='method name')
    
    # experiment
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    
    # DLI
    parser.add_argument('--dli', action='store_true', help='label fusion every')
    parser.add_argument('--poisson_prob', type=float, default=0.5, help='is poisson')
    
    # data
    parser.add_argument('--data_perc', type=float, default=1.0, help='percentage of the dataset to use')    
    
    # checkpoint
    parser.add_argument('--load_weights', type=str, help='weights file to load')
    parser.add_argument('--log_every', type=int, default=1, help='log every')

    args = parser.parse_args()
    
    return args