import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging
import os
from tensorboardX import SummaryWriter
import utils
import datetime
import os
import getData
import random
import spectral as spy
import time
import cv2

def train(model, optim, sche, opt):
    """
    Args:
        model: the model to be trained
        optim: pytorch optimizer to be used
        opt: command line input from the user
    """

    # save the training log and the loss graph
    now_time = datetime.datetime.now()
    logname = opt.save_log + opt.exp_name + str(now_time) + "log.txt"
    writer = SummaryWriter(opt.save_loss + str(now_time) + '/')
    
    # make dirction
    if not os.path.exists(opt.save_loss):
        os.makedirs(opt.save_loss)
    if not os.path.exists(opt.save_log):
        os.makedirs(opt.save_log)
    
    # evaluate model
    psnr_max = evaluate(logname, model, 0, opt)

    # set loss function
    loss_function = nn.MSELoss(reduction='mean')
        
    # start training
    for epoch in range(1, opt.epochs + 1):

        # set the model in the training mode
        model.train()

        hdf5_Idx = [0, 1, 2, 3, 4]
        for h5idx in range(opt.train_len):

            # data loader
            train_set = getData.TrainSet(hdf5_Idx[h5idx], opt)
            train_loader = torch.utils.data.DataLoader(train_set, 
                                                    batch_size = opt.batch_size, 
                                                    shuffle = True)    

            for batch_idx, batch in enumerate(train_loader):

                data = batch
                
                if opt.cuda:
                    with torch.no_grad():
                        # move to GPU
                        data = data.cuda()               

                # erase all computed gradient        
                optim.zero_grad()
                
                # forward pass to get prediction
                if batch_idx == 0:
                    gts, hsi_pred = model(data, opt.mode, training = True)
                    print('update DOE! Epoch:   ' + str(epoch) + '   h5idx:   ' + str(h5idx))
                else:
                    gts, hsi_pred = model(data, opt.mode, training = False)
                    
                loss=loss_function(hsi_pred, gts)

                # save the loss info
                writer.add_scalar("loss",loss, (epoch * opt.train_len + h5idx) * opt.batch_num + batch_idx)

                # compute gradient in the computational graph
                loss.backward(retain_graph=False)   # True
                
                # update parameters in the model 
                optim.step()

                # update learning rate
                sche.step()

                # logging
                if batch_idx % opt.report_every == 0:
                    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                        epoch, batch_idx * opt.batch_size, len(train_set),
                        100. * batch_idx / len(train_loader), loss.data.item()))

                # evaluate model
                if batch_idx % opt.test_every == (opt.test_every - 1):
                    psnr_tmp = evaluate(logname,model,epoch, opt)
                    if psnr_tmp > psnr_max:
                        utils.save_model(model, opt)
                        psnr_max = psnr_tmp

        print(model.codednet.doe.doe_height)

    logging.info('Training finished.')
    writer.close()
    return model

def evaluate(logname,model,epoch, opt):
    
    # set the model in the evaluating mode
    model.eval()
    if opt.cuda:
        model.cuda()

    PSNR = 0.
    psnr_cache = []
    SSIM = 0.
    ssim_cache = []
    LPIPS = 0.
    lpips_cache = []

    # test one .h5 file by one
    for h5idx in range(opt.valid_len):

        # data loader
        eval_set = getData.ValidSet(h5idx, opt)
        loader = torch.utils.data.DataLoader(eval_set, 
                                            batch_size = opt.batch_size, 
                                            shuffle=False)
        
        for batch_idx, batch in enumerate(loader):

            data = batch
            with torch.no_grad():
                if opt.cuda:
                    data = data.cuda()             
                gts, hsi_pred = model(data, opt.mode, training = False) 
                gts, hsi_pred = np.array(gts.transpose(3, 1).transpose(2, 1).cpu()), np.array(hsi_pred.transpose(3, 1).transpose(2, 1).cpu())

                # Calculate params and FLOPs
                # flops1 = FlopCountAnalysis(model, (data, batch_idx, 9, None, False))
                # flops2 = FlopCountAnalysis(model.codednet, (data, batch_idx, 9, None, False))
                # n_param = sum([p.nelement() for p in model.reconnet.parameters()])
                # print(f'GMac:{(flops1.total() - flops2.total()) / (1024 * 1024 * 1024)}')
                # print(f'Params:{n_param}')

                #calculate psnr
                for i in range(len(gts)):
                    psnr_cache.append(utils.Cal_PSNR_by_gt(gts[i],hsi_pred[i]))
                    ssim_cache.append(utils.Cal_SSIM(gts[i],hsi_pred[i]))
                    lpips_cache.append(utils.Cal_LPIPS(gts[i],hsi_pred[i]))

    # calculate the average PSNR SSIM LPIPS of the total validset
    PSNR = sum(psnr_cache)/len(psnr_cache)
    SSIM = sum(ssim_cache)/len(ssim_cache)
    LPIPS = sum(lpips_cache)/len(lpips_cache)

    # record the results
    logging.info(' Average_PSNR: {:.4f}. '.format(PSNR))  
    logging.info(' Average_SSIM: {:.4f}. '.format(SSIM))  
    logging.info(' Average_LPIPS: {:.4f}. '.format(LPIPS))  
    f = open(logname,'a')
    f.write('Epoch:')
    f.write(str(str(epoch)))
    f.write('      ')
    f.write(str(PSNR))
    f.write('   SSIM:   ')
    f.write(str(SSIM))
    f.write('   LPIPS:   ')
    f.write(str(LPIPS))
    f.write('\r\n')
    f.close

    return PSNR