import argparse



def str2bool(v):
    
    """ 
    Function to convert input string from argparse to bool
    
    input: v -> string
    output: v -> bool
    """
    
    if isinstance(v, bool):
        return v
    elif v.lower() == 'true':
        return True
    else:
        return False


def get_args():
    parser = argparse.ArgumentParser(description='Arguements for training')

    # Input parameters
    parser.add_argument('--model_name')
    parser.add_argument('--early_stop', default=False, type=str2bool)
    parser.add_argument('--memorize', default=False, type=str2bool)
    parser.add_argument('--easy', default=False, type=str2bool)
    parser.add_argument('--combined', default=False, type=str2bool)
    parser.add_argument('--num_linear_layers', default=2, type=int)
    parser.add_argument('--dataset', default='cifar100')
    parser.add_argument('--output_size', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--temp_str', default='', type=str)

    args = parser.parse_args()
    return args