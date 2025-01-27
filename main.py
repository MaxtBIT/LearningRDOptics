# public libraries
import logging
import time
import torch
import os

# parse command line input
import utils
opt = utils.parse_arg()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpuid)

from pytorch_lightning import  seed_everything
seed_everything(opt.seed, workers=True)

# train for baseline
import train.model as model_baseline # model implementation
import train.trainer as trainer_baseline # training functions
import optimizer # optimization functions

def main():

    # logging configuration
    logging.basicConfig(level = logging.INFO,
        format = "[%(asctime)s]: %(message)s"
    )
        
    opt.cuda = opt.gpuid>=0

    # record the current time
    opt.save_dir += time.asctime(time.localtime(time.time()))
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # initialize the model
    if opt.mode == 'baseline':
        model_train = model_baseline.prepare_model(opt)
        # loading the pre-train model (if need)
        if opt.pretrained:
            print('Loading the pre-train model ...')
            load_log = model_train.load_state_dict(torch.load(opt.pretrained_path, map_location=torch.device('cuda')), strict=False)
            # loaded_fixed_params = torch.load(opt.fixed_params_path)
            # model_train.codednet.doe_noise_1 = loaded_fixed_params['doe_noise_1']
            # model_train.codednet.doe_noise_2 = loaded_fixed_params['doe_noise_2']
            # model_train.codednet.doe.radius_doe = loaded_fixed_params['radius_doe']
            # model_train.codednet.doe.radius_doe_save = loaded_fixed_params['radius_doe_save']
            print(load_log)
    else:
        raise NotImplementedError
    
    # configurate the optimizer and learning rate scheduler
    optim, sche = optimizer.prepare_optim(model_train, opt)

    # train the model
    if opt.mode == 'baseline':
        model_train = trainer_baseline.train(model_train, optim, sche, opt)
    else:
        raise NotImplementedError

    # save the final trained model
    utils.save_model(model_train, opt)
    
    return 

if __name__ == '__main__':
    main()
