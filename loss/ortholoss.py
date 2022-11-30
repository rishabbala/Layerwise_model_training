
def ComputeOrthogonalLoss():

    ortho_loss = 0
    num_keys = 0
    for key, values in model.named_parameters():
        num_keys += 1
        
        if 'downsample.conv.weight' in key:

            c = values.reshape(values.shape[0], -1)
            # c = c/(1e-5 + torch.sqrt(torch.sum(torch.square(c.detach()), 1, keepdim=True)))
            if c.shape[0] < c.shape[1]:
                c = c.permute(1, 0)

            ortho_loss += torch.norm( torch.t(c)@c - torch.eye(c.shape[1]).to(device))#/(c.shape[0]*c.shape[1])

        elif 'conv' in key and 'weight' in key:
            # kernel = values/(1e-5 + torch.sqrt(torch.sum(torch.square(values.detach()), (2, 3), keepdim=True)))
            
            txt = key.split('.')
            
            try:
                ## layer 0 has no stride
                if int(txt[1][-1]) > 0 and int(txt[2]) == 0:
                    s = 2
                else:
                    s = 1

            except:
                continue

            p = math.floor((values.data.shape[2]-1)/s) * s

            c = torch.nn.functional.conv2d(values, values, padding=1, stride=s)
            target = torch.zeros(c.shape).to(device)
            target[:, :, int(math.floor(c.shape[2]/2)), int(math.floor(c.shape[2]/2))] = torch.eye(c.shape[0])
            ortho_loss += torch.norm(c - target)#/(c.shape[0]*c.shape[1]*c.shape[2]*c.shape[3])

    # num blocks
    if 'resnet18' in args.model_name:
        div = 8
    elif 'resnet34' in args.model_name:
        div = 16

    return ortho_loss/div