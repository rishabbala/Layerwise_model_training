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
    parser.add_argument('--num_linear_layers', default=2, type=int)
    parser.add_argument('--use_torch_model', default=False, type=str2bool)
    parser.add_argument('--dataset', default='cifar100')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--share_weights', default=False, type=str2bool)
    parser.add_argument('--train_solo_layers', default=False, type=str2bool)
    parser.add_argument('--block_grad', default=False, type=str2bool)
    parser.add_argument('--train_upper_layers', default=True, type=str2bool)
    parser.add_argument('--train_lin_layers', default=True, type=str2bool)
    parser.add_argument('--epochs_downstream', default=False, type=str2bool)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):

    if args.use_torch_model == True and (args.share_weights == True or args.train_solo_layers == True or args.block_grad == True or args.train_upper_layers == False or args.train_lin_layers == False):
        raise ValueError("Torch model can only be trained without any additional methods")
    
    if args.share_weights == False and (args.train_solo_layers == True or args.block_grad == True or args.train_upper_layers == False or args.train_lin_layers == False):
        raise ValueError("Additional methods without weight sharing is not allowed")

    if args.train_solo_layers == False and (args.block_grad == True or args.train_upper_layers == False or args.train_lin_layers == False):
        raise ValueError("Varying training method without solo layer training is not allowed")

    if args.block_grad == False and (args.train_upper_layers == False or args.train_lin_layers == False):
        raise ValueError("Must train upper layers if gradient is unblocked")