from .resnet import MakeResnet
from .vit import ViT
from .cct import CCT
from .vit_pool import ViTPool
from .vgg import MakeVGG
import math



def get_block_size(args):
    '''
        Get the model sizes for each model architecture
    '''

    block = None

    if args.combined:
        if 'resnet18' in args.model_name:
            block = [[1, 1, 1, 1], [2, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2, 1], [2, 2, 2, 2]]
        if 'resnet34' in args.model_name:
            block = [[1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1, 1], [3, 2, 1, 1], [3, 3, 1, 1], [3, 4, 1, 1], [3, 4, 2, 1], [3, 4, 3, 1], [3, 4, 4, 1], [3, 4, 5, 1], [3, 4, 6, 1], [3, 4, 6, 2], [3, 4, 6, 3]]
        if 'cct_2' in args.model_name:
            block = [[1], [2]]
        if 'cct_4' in args.model_name:
            block = [[1], [2], [3], [4]]
        if 'cct_7' in args.model_name:
            block = [[1], [2], [3], [4], [5], [6], [7]]
        if 'vit' in args.model_name:
            block = [[1], [2], [3], [4], [5], [6], [7]]
        if 'vit_pool' in args.model_name:
            block = [[1], [2], [3], [4], [5], [6], [7]]

    else:
        if 'resnet18' in args.model_name:
            block = [[2, 2, 2, 2]]
        if 'resnet34' in args.model_name:
            block = [[3, 4, 6, 3]]
        if 'cct_2' in args.model_name:
            block = [[2]]
        if 'cct_4' in args.model_name:
            block = [[4]]
        if 'cct_7' in args.model_name:
            block = [[7]]
        if 'vit' in args.model_name:
            block = [[7]]
        if 'vit_pool' in args.model_name:
            block = [[7]]
        

    if block == None:
        raise ValueError("Model name incorrect")

    if args.combined:
        num_epochs_per_block = math.floor(args.n_epochs / (2*(len(block)-1)) )
    else:
        num_epochs_per_block = args.n_epochs

    return block, num_epochs_per_block


def get_model_func(args):
    '''
    Get the function that creates the required model
    '''

    if 'vgg' in args.model_name:
        func = MakeVGG
    elif 'resnet' in args.model_name:
        func = MakeResnet
    elif 'vit' in args.model_name:
        func = ViT
    elif 'cct' in args.model_name:
        func = CCT

    return func