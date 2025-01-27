import argparse
import time
import numpy as np
import torch
from torch import save
import torchvision.transforms.functional as ttf
import torch.nn.functional as F
from os import path as path
import spectral as spy
import cv2

# function for parsing input arguments
def parse_arg():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-mode', type=str, default="baseline", help="baseline: random mask, optim: learnable mask")
    parser.add_argument('-exp_name', type=str, default="A324", help="exp_name")
    # training settings
    parser.add_argument('-gpuid', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=1)    # 16
    parser.add_argument('-epochs', type=int, default=60)
    parser.add_argument('-batch_num', type=int, default=800)  # the number of batches for each .h5 file 50
    parser.add_argument('-report_every', type=int, default=200)   # report log after x batches 50
    parser.add_argument('-test_every', type=int, default=800)  # evaluate model after x batches 50
    parser.add_argument('-train_len', type=int, default=5) # get the number of .h5 files
    parser.add_argument('-valid_len', type=int, default=1)
    # paths
    parser.add_argument('-save_dir', type=str, default='./model/')
    parser.add_argument('-save_log', type=str, default='./data_log/')
    parser.add_argument('-save_loss', type=str, default='./graph_log/')
    parser.add_argument('-pretrained_path', type=str, default='./model/div2k_retrain/trained_modelA324.pth') # div2k_retrain 4lvl_0427_loadpretrain set the path of the pre-trained model ../root/autodl-tmp/pt_model/trained_modelA324.pth
    parser.add_argument('-fixed_params_path', type=str, default='./model/div2k_retrain/trained_modelA324_nonl.pth') 
    parser.add_argument('-trainset_path', type=str, default='../root/autodl-tmp/data/') # set the path of the trainset
    parser.add_argument('-testset_path', type=str, default='../root/autodl-tmp/data/') # set the path of the testset
    # model settings
    parser.add_argument('-save_name', type=str, default='trained_model')   
    parser.add_argument('-pretrained', type=bool, default=True) # used for loading the pre-trained model
    # optimizer settings
    parser.add_argument('-optim_type', type=str, default='adam', help="adam or sgd")
    parser.add_argument('-doe_grad', type=bool, default=True)
    parser.add_argument('-lr_doe', type=float, default=0.0000000001, help="0.0000000003")  # 0.00000001 larger 0.00000000001 smaller  0.000000001 / 0.0000000005 / 0.0000000002 work
    parser.add_argument('-lr_unet', type=float, default=0.0001, help="adam:0.0001, sgd: 0.05")    #0.00001 0.00005 work
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-momentum', type=float, default=0.9, help="sgd: 0.9")
    # reduce the learning rate after each milestone
    parser.add_argument('-milestones', type=list, default=[20,40])
    # how much to reduce the learning rate
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-seed', type=int, default=2024)   # 1993 2024 34.3 40epo  worse: 1 0 4202 2025 1234 888 2019

    opt = parser.parse_args()
    return opt

# smallest positive float number
FLT_MIN = float(np.finfo(np.float32).eps)

# the function for model saving
def get_save_dir(opt, str_type=None):

    root = opt.save_dir
    save_name = path.join(root, opt.save_name + opt.exp_name) 
    # save_name += '_'
    # save_name += time.asctime(time.localtime(time.time()))
    save_name += '.pth'

    save_nonl_name = path.join(root, opt.save_name + opt.exp_name + '_nonl') 
    save_nonl_name += '.pth'

    return save_name, save_nonl_name

def save_model(model, opt):

    save_name, save_nonl_name = get_save_dir(opt)
    save(model.state_dict(), save_name)

    fixed_params = {'doe_noise_1': model.codednet.doe_noise_1,
                    'doe_noise_2': model.codednet.doe_noise_2,
                    'radius_doe': model.codednet.doe.radius_doe,
                    'radius_doe_save': model.codednet.doe.radius_doe_save
                    }
    save(fixed_params, save_nonl_name)

    return

def Cal_mse(im1, im2):

    return np.mean(np.square(im1 - im2), dtype=np.float64)

# calculate the peak signal-to-noise ratio (PSNR)
def Cal_PSNR_by_gt(im_true, im_test):

    channel  = im_true.shape[2]
    im_true  = 255*im_true
    im_test  = 255*im_test
    
    psnr_sum = 0
    for i in range(channel):
        band_true = np.squeeze(im_true[:,:,i])
        band_test = np.squeeze(im_test[:,:,i])
        err       = Cal_mse(band_true, band_test)
        max_value = np.max(np.max(band_true))
        psnr_sum  = psnr_sum+10 * np.log10((max_value ** 2) / err + FLT_MIN)
    
    return psnr_sum/channel

def Cal_PSNR_by_default(im_true, im_test):

    channel  = im_true.shape[2]   
    psnr_sum = 0.
    for i in range(channel):
        band_true = np.squeeze(im_true[:,:,i])
        band_test = np.squeeze(im_test[:,:,i])
        err       = Cal_mse(band_true, band_test)
        psnr_sum  = psnr_sum+10 * np.log10(1.0 / err)
    
    return psnr_sum/channel

# calculate the structural similarity (SSIM)
# cited by https://github.com/cszn
def Cal_SSIM(im_true, im_test):

    if not im_true.shape == im_test.shape:
        raise ValueError('Input images must have the same dimensions.')

    channel  = im_true.shape[2]
    im_true  = 255*im_true
    im_test  = 255*im_test
    ssim_sum = 0.

    for k in range(channel):
        ssim_sum = ssim_sum + ssim(im_true[:, :, k], im_test[:, :, k])

    return ssim_sum / channel

def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# calculate the spectral angle mapping (SAM)
def Cal_SAM(im_true, im_test):

    a = sum(im_true * im_test, 2) + FLT_MIN
    b = pow(sum(im_true * im_true, 2) + FLT_MIN, 1/2)
    c = pow(sum(im_test * im_test, 2) + FLT_MIN, 1/2)
    d = np.arccos(a/(b * c))

    return np.mean(d)

def get_intensities(input_field):
    return torch.square(torch.abs(input_field))

def crop_psf(psf):
    # max_y = torch.max(psf[:, 22:66, 22:66], 1)[1][:, 21, :] + 22    # max_y = torch.max(psf, 1)[1][:, 21, :]
    # cropped_psf = psf[:, (max_y - 22):(max_y + 22), 22:66, :]     # cropped_psf = psf[:, (max_y - 11):(max_y + 11), 11:33, :]

    cropped_psf = psf[:, 22:66, 22:66, :]

    return cropped_psf

def circular_aperture(input_field):
    input_shape = input_field.shape
    [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2, -input_shape[2] // 2: input_shape[2] // 2].astype(
        np.float64)
    max_val = np.amax(x)
    r = np.sqrt(x ** 2 + y ** 2)[None, :, :, None]

    aperture = (r < max_val).astype(np.float64)
    aperture = torch.Tensor(aperture).cuda()

    return aperture

def circular_aperture_for_show(input_field):
    input_shape = input_field.shape
    [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2, -input_shape[2] // 2: input_shape[2] // 2].astype(
        np.float64)
    max_val = np.amax(x)-4.0
    r = np.sqrt(x ** 2 + y ** 2)[None, :, :, None]

    aperture = (r < max_val).astype(np.float64)
    aperture = torch.Tensor(aperture).cuda()

    return aperture

def img2patch(img, row_num, col_num, patch_size, psf_size):

    pad_width=psf_size // 2
    img_pad = F.pad(img, pad=(pad_width, pad_width, pad_width, pad_width, 0, 0), mode='constant')
    center_axis = []
    for i in range(row_num):
        for j in range(col_num):
            row_axis = patch_size // 2 + psf_size // 2 + i * 64
            col_axis = patch_size // 2 + psf_size // 2 + j * 64
            center_axis.append([row_axis, col_axis])

    patch_list = []
    for i in range(len(center_axis)):
        row_cen = center_axis[i][0]
        col_cen = center_axis[i][1]
        half_seg_size = patch_size // 2 + psf_size // 2
        patch = img_pad[:, (row_cen - half_seg_size):(row_cen + half_seg_size), (col_cen - half_seg_size):(col_cen + half_seg_size)]
        patch_list.append(patch)

    return patch_list, center_axis

def patch_conv(patch_list, psf_list, psf_index, angle_list):

    conv_patch_list = []
    for i in range(len(patch_list)):
        img = patch_list[i]
        psf = psf_list[psf_index[i]]
        psf_ = ttf.rotate(psf, angle_list[i])
        normalized_psf = psf_ / torch.sum(psf_, dim=(1,2), keepdim=True)
        conv_patch = [torch.conv2d(img[c,...].unsqueeze(0).unsqueeze(1), normalized_psf[c,...].unsqueeze(0).unsqueeze(1)) for c in range(7)]
        conv_patch = torch.cat(conv_patch, 1)
        conv_patch_list.append(conv_patch[0, :, 0:64, 0:64])

    return conv_patch_list

def patch2img(conv_patch_list):

    conv_img_list = []
    for i in range(9):
        conv_img_row = torch.cat(conv_patch_list[10 * i:10 * (i+1)], 2)
        conv_img_list.append(conv_img_row)
    conv_img = torch.cat(conv_img_list, 1)

    return conv_img
