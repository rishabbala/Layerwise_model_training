import torch
import os
from resnet import MakeResnet
import shutil
import math



def CreateTorchModel(args, output_size):
    '''
    Create models fro torch zoo

    input: 
            args: the arguments -> dict
            output_size: the final output size of the network -> int

    output:
            parameters: dictionary containing model (the model) and optim_parameters (the model.parameters()) -> dict
    '''
    if args.use_torch_model == True:
        if output_size < 512:
            num_features_per_lin_layer = int((512-output_size)/args.num_linear_layers)
            lin = [nn.Flatten(start_dim=1)]
            for i in range(args.num_linear_layers-1):
                lin.append(nn.Linear(in_features=512-i*num_features_per_lin_layer, out_features=512-(i+1)*num_features_per_lin_layer))
                lin.append(nn.ReLU(inplace=True))
                final_size = 512-(i+1)*num_features_per_lin_layer
        else:
            num_features_per_lin_layer = int((output_size-512)/args.num_linear_layers)
            lin = [nn.Flatten(start_dim=1)]
            for i in range(args.num_linear_layers-1):
                lin.append(nn.Linear(in_features=512+i*num_features_per_lin_layer, out_features=512+(i+1)*num_features_per_lin_layer))
                lin.append(nn.ReLU(inplace=True))
                final_size = 512+(i+1)*num_features_per_lin_layer

        lin.append(nn.Linear(in_features=final_size, out_features=output_size))
        lin = nn.Sequential(*lin)

    if "resnet18" in args.model_name:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model = nn.Sequential(model, lin)
    
    if "resnet18" in args.model_name:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model = nn.Sequential(model, lin)

    optim_parameters = model.parameters()
    parameters = {'model': model,
                  'optim_parameters': optim_parameters} 

    return parameters



def CreateCustomModel(args, output_size):
    '''
    Create models implemented here

    input: 
            args: the arguments -> dict
            output_size: the final output size of the network -> int

    output:
            parameters -> dict:
                                model (the model)
                                optim_parameters (the parameters to be optimized) 
                                num_layers_below (the number of layers that come after the current block, used for computing the number of epochs for each block when training solo layers)
                                base_size (size in layers per block of the current network)
                                layer_increase (number of layers increased per block from the smaller model)
                                share_pos (current block position till which the intermediate models are trained)
                                num_epochs_per_each_block (the number of training epochs for each block)
    '''

    # If the whole model is to be trained or a selected set of layers are to be trained together
    if not args.train_solo_layers:
        if "resnet18" in args.model_name:
            model = MakeResnet('resnet18', args.num_linear_layers, output_size, args.dataset)
            optim_parameters = model.parameters()
            if args.share_weights == True:
                raise ValueError("Cannot perform shared weight training for base model")

        elif "resnet34" in args.model_name:
            model = MakeResnet('resnet34', args.num_linear_layers, output_size, args.dataset)
            optim_parameters = model.parameters()
            if args.share_weights == True:
                model, optim_parameters = WeightShare(args, model, './weights/'+str(args.dataset)+'/resnet18/model.pth')

        num_layers_below = []
        share_pos = 0
        base_size = []
        layer_increase = []
        num_epochs_per_each_block = []

    # If the blocks are to be trained sequentially
    else:
        if "resnet18" in args.model_name:
            raise NotImplementedError("Solo layer training for base models not implemented")
        elif 'resnet34' in args.model_name:
            # Assuming that the base model is resnet18
            num_layers_below = [8, 6, 4, 2]
            share_pos = -1
            base_size = [3, 2, 2, 2]
            layer_increase = [1, 2, 4, 1]
            
            if not args.train_upper_layers:
                num_new_layers = 8
            else:
                if not args.epochs_downstream:
                    num_new_layers = 8
                else:
                    print("Downstream")
                    num_train_layers = list(map(lambda a,b: a+b, num_layers_below, layer_increase))
                    num_new_layers = sum(num_train_layers)
            
            model, optim_parameters = CreateBlockwiseResnet(args, base_size, output_size, './weights/'+str(args.dataset)+'/resnet18/model.pth', 0)

        num_epochs_per_each_block = SplitEpochs(args, num_layers_below, base_size, layer_increase, share_pos, num_new_layers)

    parameters = {'model': model,
                  'optim_parameters': optim_parameters,
                  'num_layers_below': num_layers_below,
                  'base_size': base_size,
                  'layer_increase': layer_increase,
                  'share_pos': share_pos,
                  'num_epochs_per_each_block': num_epochs_per_each_block} 

    return parameters



def CreateBlockwiseResnet(args, other_size, output_size, weight_share_dir, block_number):

    """ 
    If training blockwise, we build the model block by block and train them seperately

    input: 
            args: arguments
            other_size: a list containing the size of each block in the resnet -> list
            output_size: output_size of network -> int
            weight_share_dir: the directory where weights of smaller model or the one without the current block is present -> str
            block_number: current block position where the new layers are added 
    output:
            model: the model to train -> torch model
            optim_parameters: subset of model parameters to optimize -> list
    """

    model = MakeResnet('other', args.num_linear_layers, output_size, args. dataset, other_size=other_size)
    model, optim_parameters = WeightShare(args, model, weight_share_dir, block_number)

    return model, optim_parameters



def WeightShare(args, model, child_weight_path, block_number=None):

    """ 
    Function to share weights between smaller and larger models

    input:
            args: arguments passed to the program -> from argparse
            model: the model to be trained -> torch model
            child_weight_path: dir of weights of a smaller model -> str
            block_number: the block number to which new layers are added so we can freeze the layers above it and unfreeze those below it -> int

    output:
            model: the model to be trained -> torch model
            optim_parameters: subset of model parameters to optimize -> list
    """

    optim_parameters = []
    linear_layer_values = []

    ## Load all previous layers with weight from smaller model
    sd = model.state_dict()
    try:
        child_sd = torch.load(child_weight_path)
        sd.update(child_sd)
    except:
        raise ValueError("Base model not trained yet")
    model.load_state_dict(sd)

    ## Initialize just the weights and bias ONLY FOR RESNETS, MAKE SURE TO CHANGE FOR OTHERS. bias comes after weight so change just the layer jsut after weight
    for key, values in model.named_parameters():
        if key not in child_sd.keys():

            # weight of conv layer
            if 'weight' in key and len(list(values.shape)) > 1:
                kernel = torch.zeros(values.shape[2], values.shape[3])
                # kernel[int(values.shape[2]/2), int(values.shape[3]/2)] = 1
                kernel.unsqueeze(0).unsqueeze(0)
                kernel = torch.tile(kernel, (values.shape[0], values.shape[1], 1, 1))
                values.data = kernel

            # bias is 0, weight of batchnorm is 1
            else:
                if 'weight' in key:
                    values.data = torch.ones(values.shape)
                else:
                    values.data = torch.zeros(values.shape)

    # The parameters to be trained are chosen and the others have their requires_grad set to False. Make sure downstream layers are are added to optimizer. This is done using the model nomenclature. We know that layer1 is 2 convs for resnet, which is always frozen. Layer 2 is the first block. So we check from layer 2 onwards. print(model.named_parameters()) to verify
    if args.block_grad:

        if args.train_lin_layers == False:
            func = lambda key: False if key in child_sd else True
        elif args.train_lin_layers == True:
            func = lambda key: False if key in child_sd and 'linear_layers' not in key else True

        for (key, values) in model.named_parameters():
            values.requires_grad = func(key)
            txt = key.split(".")
            try:
                t = int(txt[1][-1])
                if (t > block_number+2 or values.requires_grad == True) and not args.train_upper_layers:
                    optim_parameters.append(values)
                elif t > block_number+2 and args.train_upper_layers:
                    values.requires_grad = True
                    optim_parameters.append(values)
            
            ## Linear layers come here because of the naming used when creating our model
            except:
                if values.requires_grad == True:
                    optim_parameters.append(values)
    else:
        optim_parameters = model.parameters()
        
    return model, optim_parameters



def SplitEpochs(args, num_layers_below, base_size, layer_increase, share_pos, num_new_layers):

    num_epochs_per_each_block = []
    temp = 0
    for i in range(len(layer_increase)):
        if not args.epochs_downstream:
            num_epochs_per_each_block.append(math.ceil(layer_increase[i]*args.n_epochs/num_new_layers))
            temp += math.ceil(layer_increase[i]*args.n_epochs/num_new_layers)
        else:
            if not args.train_upper_layers:
                num_epochs_per_each_block.append(math.ceil(layer_increase[i]*args.n_epochs/num_new_layers))
                temp += math.ceil(layer_increase[i]*args.n_epochs/num_new_layers)
            else:
                num_epochs_per_each_block.append(math.ceil(num_train_layers[i]*args.n_epochs/num_new_layers))
                temp += math.ceil(num_train_layers[i]*args.n_epochs/num_new_layers)

    args.n_epochs = temp

    return num_epochs_per_each_block