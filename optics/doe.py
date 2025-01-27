import torch
import torch.nn as nn
import numpy as np
from .util import complex_exponent_tf
import spectral as spy
from utils import circular_aperture_for_show
from .binary import *

def radius_to_height_map_with_block(height_map_1d, block_size,index,over_index):

    height_map = height_map_1d[index]  # 一维度数据经过2维索引就变成了2维数据
    height_map[over_index==1] = 0  # 超出半径的置为0
        
    height_map_half_left = torch.cat([torch.flip(height_map, dims=[0]), height_map], axis=0)
    height_map_full = torch.cat([torch.flip(height_map_half_left, dims=[1]), height_map_half_left], axis=1)

    return height_map_full.unsqueeze(0).unsqueeze(3)

def gen_radius(rad_idx_b1, rad_idx_b2, doe_height, doe_noise_1, doe_noise_2):

    b1 = binarize(rad_idx_b1)
    b2 = binarize(rad_idx_b2)
    radius_idx = 1e-3 - b1 * doe_height[0] * doe_noise_1 - b2 * doe_height[1] * doe_noise_2

    return radius_idx

class DOE(nn.Module):
    def __init__(self, field_shape, grad_doe):
        super(DOE, self).__init__()
        self.block_size= 4
        self.doe_height = nn.Parameter(torch.FloatTensor([150e-9, 250e-9]).cuda(), requires_grad = grad_doe)   # 550e- hao 150 350   buhao 200 400  guding 150e-9, 350e-9youhuabuchu 200e-9, 400e-9gengcha
        self.rad_idx_b1 = nn.Parameter(torch.randn(size =  [field_shape[1] // (2 * self.block_size)]).cuda() / 1e2, requires_grad = grad_doe) 
        self.rad_idx_b2 = nn.Parameter(torch.randn(size =  [field_shape[1] // (2 * self.block_size)]).cuda() / 1e2, requires_grad = grad_doe)

        doe_noise_1 = torch.rand(size = [256]).cuda() * 0.10 + 0.95
        doe_noise_2 = torch.rand(size = [256]).cuda() * 0.10 + 0.95
        self.radius_doe = gen_radius(self.rad_idx_b1, self.rad_idx_b2, self.doe_height, doe_noise_1, doe_noise_2)
        self.doe_no_noise_save = torch.rand(size = [256]).cuda() * 0.0 + 1.0
        self.radius_doe_save = gen_radius(self.rad_idx_b1, self.rad_idx_b2, self.doe_height, self.doe_no_noise_save, self.doe_no_noise_save).detach()

        radius = field_shape[1] // (2 * self.block_size)
        block_size = self.block_size
        [x, y] = torch.tensor(np.mgrid[0:radius*block_size, 0:radius*block_size], dtype=torch.float32)
        radius_distance = (torch.sqrt(x ** 2 + y ** 2)/block_size).cuda()
        index = torch.floor(radius_distance).long()  # 对应1D的位置索引,向下取整
        over_index = torch.zeros(size=index.shape,dtype=torch.int)  # 超出半径边界的索引位置
        over_index[index>=radius] = 1
        index[over_index==1] = 0
        self.index = index
        self.over_index = over_index

        self.weight_height_map = radius_to_height_map_with_block(self.radius_doe, self.block_size,self.index,self.over_index)


    def modulate(self, input_field, delta_n, wave_lengths, doe_noise_1, doe_noise_2):

        self.radius_doe = gen_radius(self.rad_idx_b1, self.rad_idx_b2, self.doe_height, doe_noise_1, doe_noise_2)
        self.weight_height_map = radius_to_height_map_with_block(self.radius_doe, self.block_size, self.index, self.over_index)

        self.radius_doe_save = gen_radius(self.rad_idx_b1, self.rad_idx_b2, self.doe_height, self.doe_no_noise_save, self.doe_no_noise_save).detach()
        self.weight_height_map_no_noise = radius_to_height_map_with_block(self.radius_doe_save, self.block_size, self.index, self.over_index)
        # print(self.doe_height)
        
        # ========================
        # doe_params = np.array(self.radius_doe_save.detach().cpu())
        # f = open('./doe_params.txt','w')
        # for i in range(doe_params.shape[0]):
        #     f.write(str(doe_params[i]))
        #     f.write('\r\n')
        # f.close
        # ========================
        # doe_min = torch.amin(self.radius_doe.cpu().detach())
        # doe_max = torch.amax(self.radius_doe.cpu().detach())
        # doe_show = self.weight_height_map.cpu().detach()
        # aperture = circular_aperture_for_show(doe_show)
        # doe_show = (doe_show - doe_min) / (doe_max - doe_min) * aperture.cpu()
        # spy.save_rgb('./doe.png', np.array(torch.tile(doe_show[0,...], (1, 1, 3))))
        doe_min = torch.amin(self.radius_doe_save.cpu().detach())
        doe_max = torch.amax(self.radius_doe_save.cpu().detach())
        doe_show = self.weight_height_map_no_noise.cpu().detach()
        aperture = circular_aperture_for_show(doe_show)
        doe_show = (doe_show - doe_min) / (doe_max - doe_min) * aperture.cpu()
        spy.save_rgb('./doe.png', np.array(torch.tile(doe_show[0,...], (1, 1, 3))))
        # ========================

        wave_numbers = 2. * torch.pi / wave_lengths
        phi = wave_numbers * (delta_n - 1.0)  * self.weight_height_map
        phase_shifts = complex_exponent_tf(phi)
        shifted_field = torch.mul(input_field, phase_shifts)
        
        return shifted_field