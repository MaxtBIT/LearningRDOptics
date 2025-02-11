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

def test_real(model, optim, sche, opt):

    now_time = datetime.datetime.now()
    logname = opt.save_log + opt.exp_name + str(now_time) + "log.txt"

    # evaluate model
    evaluate(logname,model, 0, opt)

    logging.info('Test finished.')

    return

def evaluate(logname,model,epoch, opt):
    
    # set the model in the evaluating mode
    model.eval()
    if opt.cuda:
        model.cuda()

    files = os.listdir('./real_capture/')

    # data loader
    for file in files:

        img = cv2.imread('./real_capture/' + file)
        img = img / 255.0 * 0.96

        data = torch.tensor(img,dtype=torch.float32).unsqueeze(0).transpose(3,1).transpose(3,2)

        with torch.no_grad():
            if opt.cuda:
                data = data.cuda()
                
            _, hsi_pred = model(data, opt.mode, training = False) 

            hsi_pred[hsi_pred < 0] = 0.0
            hsi_pred[hsi_pred > 1] = 1.0

            hsi_pred[:,0,...] = hsi_pred[:,0,...] * 2.35
            hsi_pred[:,1,...] = hsi_pred[:,1,...] * 1.0
            hsi_pred[:,2,...] = hsi_pred[:,2,...] * 1.75

            data[:,0,...] = data[:,0,...] * 2.35
            data[:,1,...] = data[:,1,...] * 1.0
            data[:,2,...] = data[:,2,...] * 1.75

            hsi_pred[hsi_pred < 0] = 0.0
            hsi_pred[hsi_pred > 1] = 1.0

            data = data * 0.9
            hsi_pred = hsi_pred * 0.9

            cv2.imwrite('./real_recon/recon_' + file[:-4] + '_real.png', np.array(hsi_pred[0,...].transpose(2, 0).transpose(1, 0).cpu() * 255.0).astype(int))
            print("write")
        print(file)

    return