import torch

# set the optimizer and the scheduler
def prepare_optim(model, opt):
    params = [ p for p in model.parameters() if p.requires_grad]
    
    for i in range(len(opt.milestones)):
        opt.milestones[i] = opt.milestones[i] * opt.batch_num * opt.train_len

    # optimizer type
    if opt.optim_type == 'adam':
        if opt.doe_grad:
            print('doe_grad is TRUE!')
            optimizer = torch.optim.Adam([{'params':params[0], 'lr':opt.lr_doe, 'weight_decay':opt.weight_decay},
                                      {'params':params[1::], 'lr':opt.lr_unet, 'weight_decay':opt.weight_decay}])
        else:
            print('doe_grad is FALSE!')
            optimizer = torch.optim.Adam(params, lr = opt.lr_unet, 
                                     weight_decay = opt.weight_decay)
        
    elif opt.optim_type == 'sgd':
        optimizer = torch.optim.SGD(params, lr = opt.lr, 
                                    momentum = opt.momentum,
                                    weight_decay = opt.weight_decay)   

    # scheduler with pre-defined learning rate decay
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones = opt.milestones, 
                                                    gamma = opt.gamma)

    return optimizer, scheduler
