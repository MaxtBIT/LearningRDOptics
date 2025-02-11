# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import spectral as spy
import torchvision.transforms.functional as ttf
import torch.nn.functional as F
import cv2


from optics.propagation import FresnelPropagation_with_vector, FresnelPropagation_without_vector
from optics.doe import DOE
from optics.basics import *
from optics.util import complex_exponent_tf
import optics.do as do
import utils
from utils import get_intensities, crop_psf, circular_aperture, img2patch, patch_conv, patch2img

class CodedNet(nn.Module):

    def __init__(self):

        super(CodedNet, self).__init__()
        
        opt = utils.parse_arg()
        self.wave_size = 2048
        self.delta_air = 1.000

        # initialize lens
        device = torch.device('cuda')
        self.lens = do.Lensgroup(device=device)
        self.lens.load_file(Path('./params/lens2.txt'))

        # load doe_fields_dict aperture angle_list psf_index params_dict response_dict jgs1_dict
        self.input_fields_dict = np.load('./params/doe_fields_dict_7wls.npy' ,allow_pickle=True).item()
        example_field = self.input_fields_dict['550fov0']
        example_field = torch.tensor(example_field, dtype=torch.complex64).unsqueeze(0).unsqueeze(3).cuda()
        self.aperture = circular_aperture(example_field)
        self.angle_list, self.psf_index = np.load('./params/angle_list.npy'), np.load('./params/psf_index.npy')
        self.params_dict = np.load('./params/params_dict_7wls.npy', allow_pickle=True).item()
        self.response_dict = np.load('./params/response_7wls.npy', allow_pickle=True).item()
        self.jgs1_dict = np.load('./params/jgs1.npy', allow_pickle=True).item()

        # initialize doe
        self.doe = DOE(example_field.shape, grad_doe = opt.doe_grad)

        self.wavelengths_str = ['460', '490', '520', '550', '580', '610', '640']
        self.fovs_str = ['fov0', 'fov5', 'fov10', 'fov15', 'fov20', 'fov25', 'fov30']
        propagation_DOE2Lens_list = []
        propagation_Lens2Plane_list = []
        for wavelength in self.wavelengths_str:
            for fov in self.fovs_str:
                prop_params = self.params_dict[wavelength + fov]
                p1_loc = prop_params[0]
                p2_loc = prop_params[1]
                shift_y1 = prop_params[3]
                shift_y2 = prop_params[4]
                shift_fre = prop_params[5]
                asm_sample_rate = prop_params[6]
                wavelengths = float(wavelength) * 1e-9
                distance_DOE2Lens = 28.00 * 1e-3 + p1_loc
                distance_Lens2Plane = 39.60 * 1e-3 + p1_loc

                aperture_round = [-2.048 * 1e-3, 2.048 * 1e-3, -2.048 * 1e-3, 2.048 * 1e-3]
                p1_round = [-3.20 * 1e-3, 3.20 * 1e-3, (-3.20 + shift_y1) * 1e-3, (3.20 + shift_y1) * 1e-3]
                sensor_round = [-0.512 * 1e-3, 0.512 * 1e-3, (-0.512 + shift_y2) * 1e-3, (0.512 + shift_y2) * 1e-3]
                spatial_frenq = [-0.15, 0.15, -0.15 + shift_fre, 0.15 + shift_fre]

                propagation_DOE2Lens = FresnelPropagation_with_vector(input_shape = example_field.shape, 
                                                                    distance = distance_DOE2Lens, 
                                                                    wave_lengths = wavelengths,
                                                                    aperture_round = aperture_round,
                                                                    sensor_round = p1_round,
                                                                    spatial_frenq = spatial_frenq,  
                                                                    asm_sample_rate = asm_sample_rate)

                propagation_Lens2Plane = FresnelPropagation_without_vector(input_shape = example_field.shape, 
                                                                    distance = distance_Lens2Plane,
                                                                    wave_lengths = wavelengths, 
                                                                    aperture_round = p1_round,
                                                                    sensor_round = sensor_round,
                                                                    spatial_frenq = [-0.10, 0.10, -0.10, 0.10], 
                                                                    asm_sample_rate = 2048)

                propagation_DOE2Lens_list.append(((wavelength + fov), propagation_DOE2Lens))
                propagation_Lens2Plane_list.append(((wavelength + fov), propagation_Lens2Plane))

        self.propagation_DOE2Lens_dic = dict(propagation_DOE2Lens_list)
        self.propagation_Lens2Plane_dic = dict(propagation_Lens2Plane_list)
        self.psf_list = []
        self.psf_list_detach = []
        self.psfs = torch.zeros([90, 3, 44, 44])
        self.psfs_detach = torch.zeros([90, 3, 44, 44])
        self.doe_noise_1 = torch.rand(size = [256]).cuda() * 0.10 + 0.95 # Etching depth noise
        self.doe_noise_2 = torch.rand(size = [256]).cuda() * 0.10 + 0.95 
        self.psf_real = torch.tensor(np.load('./params/psf_real.npy')).cuda() # Calibrated PSFs

    def forward(self, x, mode, training):

        response_b, response_g, response_r = torch.tensor(self.response_dict['b']).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda(), torch.tensor(self.response_dict['g']).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda(), torch.tensor(self.response_dict['r']).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        
        if mode == 'simu':
            if training is False and len(self.psf_list) != 0:
                # Get measurement.
                measure_list = []
                gt_list = []
                for i in range(x.shape[0]):
                    gt_bgr = torch.cat([torch.sum(x[i].unsqueeze(0) * response_b, 1, keepdim=True),
                                        torch.sum(x[i].unsqueeze(0) * response_g, 1, keepdim=True),
                                        torch.sum(x[i].unsqueeze(0) * response_r, 1, keepdim=True)], 1)
                    factor = torch.max(gt_bgr)
                    gt_bgr = gt_bgr / factor
                    gt_list.append(gt_bgr)

                    patch_list, _ = img2patch(x[i,...], row_num=9, col_num=10, patch_size=64, psf_size=44)
                    conv_patch_list = patch_conv(patch_list, self.psf_list_detach, self.psf_index, self.angle_list)
                    measure_input = patch2img(conv_patch_list).unsqueeze(0)
                    meas_bgr = torch.cat([torch.sum(measure_input * response_b, 1, keepdim=True),
                                        torch.sum(measure_input * response_g, 1, keepdim=True),
                                        torch.sum(measure_input * response_r, 1, keepdim=True)], 1)
                    meas_bgr = meas_bgr / factor
                    measure_list.append(meas_bgr)

                measures = torch.cat(measure_list, 0)
                gts = torch.cat(gt_list, 0)

                return measures, gts, self.psfs_detach

            else:
                self.psf_list = []
                self.psf_list_detach = []
                psf_tmp = []
                # self.doe_noise_1 = torch.rand(size = [256]).cuda() * 0.10 + 0.95   # train
                # self.doe_noise_2 = torch.rand(size = [256]).cuda() * 0.10 + 0.95   # train
                doe_noise = torch.rand(size = [256]).cuda() * 0.0 + 1.0    # test
                for index,wavelength in enumerate(self.wavelengths_str):
                    psf_fovs = []
                    for fov in self.fovs_str:
                        wavelengths = float(wavelength) * 1e-9
                        prop_params = self.params_dict[wavelength + fov]
                        p1_loc = prop_params[0]
                        p2_loc = prop_params[1]
                        delta_glass = prop_params[2]
                        shift_y1 = prop_params[3]
                        p1_round = [-3.20 * 1e-3, 3.20 * 1e-3, (-3.20 + shift_y1) * 1e-3, (3.20 + shift_y1) * 1e-3]
                        delta_jgs1 = self.jgs1_dict[wavelength + fov]

                        x2 = np.linspace(p1_round[0], p1_round[1], self.wave_size, endpoint=True)
                        y2 = np.linspace(p1_round[2], p1_round[3], self.wave_size, endpoint=True)

                        xx2, yy2 = np.meshgrid(x2, y2, indexing='xy')
                        coordxy_o = torch.tensor(np.concatenate([np.expand_dims(xx2, axis=2), np.expand_dims(yy2, axis=2), np.expand_dims(np.ones(xx2.shape) * p1_loc, axis=2)], axis=2)).cuda()
                        coordxy_p1 = torch.tensor(np.concatenate([np.expand_dims(xx2, axis=2), np.expand_dims(yy2, axis=2), np.expand_dims(np.ones(xx2.shape) * p1_loc, axis=2)], axis=2) * 1e3).cuda()
                        coordxy_p2 = torch.tensor(np.concatenate([np.expand_dims(xx2, axis=2), np.expand_dims(yy2, axis=2), np.expand_dims(np.ones(xx2.shape) * p2_loc, axis=2)], axis=2) * 1e3).cuda()
                        coordxy0 = torch.zeros(coordxy_o.shape)

                        # 1.Modulate with DOE.
                        init_field = torch.tensor(self.input_fields_dict[wavelength + fov], dtype=torch.complex64).unsqueeze(0).unsqueeze(3).cuda()
                        field_after_height_map = self.doe.modulate(init_field, delta_jgs1, wavelengths, self.doe_noise_1, self.doe_noise_2) * self.aperture

                        # 2.Propagate to lens.
                        lens_incident_field, ray_vector = self.propagation_DOE2Lens_dic[wavelength + fov].propagate(field_after_height_map)

                        # 3.Modulate with lens.
                        with torch.no_grad():
                            ray_vector = ray_vector.squeeze(3)
                            coordxy0[:, :, 0] = coordxy_o[:, :, 0] - ray_vector[:, :, 0] / ray_vector[:, :, 2] * 10.0 * 1e-3
                            coordxy0[:, :, 1] = coordxy_o[:, :, 1] - ray_vector[:, :, 1] / ray_vector[:, :, 2] * 10.0 * 1e-3
                            coordxy0[:, :, 2] = coordxy_o[:, :, 2] - 10.0 * 1e-3
                            coordxy0 = coordxy0 * 1e3

                            ray_2 = Ray(coordxy0, ray_vector, wavelengths * 1e9, device=torch.device('cuda'))
                            _, p_list_2, _ = self.lens.trace_to_sensor(ray_2, ignore_invalid=False)

                            distance1 = torch.sqrt(torch.sum((coordxy_p1 - p_list_2[0]) ** 2, dim=2)) * 1e-3
                            distance1 = torch.where(torch.isnan(distance1), torch.zeros_like(distance1), distance1)
                            distance2 = torch.sqrt(torch.sum((p_list_2[1] - coordxy_p2) ** 2, dim=2)) * 1e-3
                            distance2 = torch.where(torch.isnan(distance2), torch.zeros_like(distance2), distance2)
                            distance3 = torch.sqrt(torch.sum((p_list_2[1] - p_list_2[0]) ** 2, dim=2)) * 1e-3
                            distance3 = torch.where(torch.isnan(distance3), torch.zeros_like(distance3), distance3)

                            wave_numbers = 2.0 * torch.pi / wavelengths
                            lens_phase = wave_numbers * ( distance3 *  delta_glass  - distance1 * self.delta_air - distance2 * self.delta_air)
                            phase_to_complex_lens_phase = complex_exponent_tf(lens_phase.unsqueeze(0).unsqueeze(3))
                        lens_output_fields = torch.mul(phase_to_complex_lens_phase.to(dtype=torch.complex64), lens_incident_field)

                        # 4.Propagate to sensor.
                        sensor_incident_field = self.propagation_Lens2Plane_dic[wavelength + fov].propagate(lens_output_fields)  

                        # 5.Get PSF and normalize it.
                        psf = crop_psf(get_intensities(sensor_incident_field))
                        normalized_psf = psf / torch.sum(psf, dim=(1,2), keepdim=True)
                        psf_fovs.append(normalized_psf)

                    psf_tmp.append(torch.cat(psf_fovs, dim=0))
                psf_tmp = torch.cat(psf_tmp, dim=3)  # 7, 44, 44, 7

                for i in range(psf_tmp.shape[0]):
                    self.psf_list.append(psf_tmp[i,...].transpose(2, 0).transpose(2, 1))
                    self.psf_list_detach.append(psf_tmp[i,...].transpose(2, 0).transpose(2, 1).detach())
                print('Compute PSFs Finished!')

                # 6.Get measurement.
                measure_list = []
                gt_list = []
                for i in range(x.shape[0]):

                    gt_bgr = torch.cat([torch.sum(x[i].unsqueeze(0) * response_b, 1, keepdim=True),
                                        torch.sum(x[i].unsqueeze(0) * response_g, 1, keepdim=True),
                                        torch.sum(x[i].unsqueeze(0) * response_r, 1, keepdim=True)], 1)
                    factor = torch.max(gt_bgr)
                    gt_bgr = gt_bgr / factor
                    gt_list.append(gt_bgr)

                    patch_list, _ = img2patch(x[i,...], row_num=9, col_num=10, patch_size=64, psf_size=44)
                    conv_patch_list = patch_conv(patch_list, self.psf_list, self.psf_index, self.angle_list)
                    measure_input = patch2img(conv_patch_list).unsqueeze(0)
                    meas_bgr = torch.cat([torch.sum(measure_input * response_b, 1, keepdim=True),
                                        torch.sum(measure_input * response_g, 1, keepdim=True),
                                        torch.sum(measure_input * response_r, 1, keepdim=True)], 1)
                    meas_bgr = meas_bgr / factor
                    measure_list.append(meas_bgr)

                measures = torch.cat(measure_list, 0)
                gts = torch.cat(gt_list, 0)

                # concat psf
                psf_tmp_list = []
                for i in range(len(patch_list)):
                    psf = self.psf_list[self.psf_index[i]]
                    psf = ttf.rotate(psf, self.angle_list[i])
                    normalized_psf = psf / torch.sum(psf, dim=(1,2), keepdim=True)
                    psf_tmp_list.append(normalized_psf.unsqueeze(0))
                self.psfs = torch.cat(psf_tmp_list, 0)
                self.psfs_detach = torch.cat(psf_tmp_list, 0).detach()

                return measures, gts, self.psfs

        elif mode == 'real':
            # Get measurement.
            for i in range(self.psf_real.shape[0]):
                self.psf_list.append(self.psf_real[i,...].transpose(2, 0).transpose(2, 1))
                self.psf_list_detach.append(self.psf_real[i,...].transpose(2, 0).transpose(2, 1).detach())
            psf_tmp_list = []
            for i in range(90):
                psf = self.psf_list[self.psf_index[i]]
                psf = ttf.rotate(psf, self.angle_list[i])
                normalized_psf = psf / torch.sum(psf, dim=(1,2), keepdim=True)
                psf_tmp_list.append(normalized_psf.unsqueeze(0))
            self.psfs_detach = torch.cat(psf_tmp_list, 0).detach()

            return x, x, self.psfs_detach
        
        else:
            raise NotImplementedError
            
def CodedModel():
    return CodedNet()