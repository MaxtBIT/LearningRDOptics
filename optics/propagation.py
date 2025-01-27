import torch
from .util import mdft, midft, midft_novec

class FresnelPropagation_with_vector():
    def __init__(self, input_shape, distance, wave_lengths, aperture_round, sensor_round, spatial_frenq, asm_sample_rate):
        super().__init__()

        self.wave_lengths = wave_lengths
        self.wave_nos = 2. * torch.pi / wave_lengths
        self.distance = distance
        input_width = input_shape[2]
        input_heigth = input_shape[1]

        # coordinates of aperture
        self.x = torch.linspace(aperture_round[0], aperture_round[1], input_width).to(dtype=torch.complex64).cuda()
        self.y = torch.linspace(aperture_round[2], aperture_round[3], input_heigth).to(dtype=torch.complex64).cuda()

        # coordinates of sensor
        self.sx = torch.linspace(sensor_round[0], sensor_round[1], input_width).to(dtype=torch.complex64).cuda()
        self.sy = torch.linspace(sensor_round[2], sensor_round[3], input_heigth).to(dtype=torch.complex64).cuda()

        # range of frequency
        fxmin = 1.0 / wave_lengths * spatial_frenq[0]
        fxmax = 1.0 / wave_lengths * spatial_frenq[1]
        fymin = 1.0 / wave_lengths * spatial_frenq[2]
        fymax = 1.0 / wave_lengths * spatial_frenq[3]

        # frequency sampling
        self.fx = torch.linspace(fxmin, fxmax, asm_sample_rate).to(dtype=torch.complex64).cuda()
        self.fy = torch.linspace(fymin, fymax, asm_sample_rate).to(dtype=torch.complex64).cuda()
        
    def propagate(self, input_field):

        input_field = input_field[0, :, :, 0]
        fxx, fyy = torch.meshgrid(self.fx, self.fy, indexing='xy')
        fxx, fyy = fxx.to(dtype=torch.complex64).cuda(), fyy.to(dtype=torch.complex64).cuda()
        H_func = torch.exp(1j * self.wave_nos * self.distance * torch.sqrt(1 - (fxx * self.wave_lengths)**2 - (fyy * self.wave_lengths)**2))

        decompose_field = mdft(input_field, self.x, self.y, self.fx, self.fy).squeeze(0)
        prop_field = decompose_field * H_func
        output_field = midft(prop_field, self.sx, self.sy, self.fx, self.fy, self.wave_lengths)

        return output_field[0].unsqueeze(3), output_field[1].detach()

class FresnelPropagation_without_vector():
    def __init__(self, input_shape, distance, wave_lengths, aperture_round, sensor_round, spatial_frenq, asm_sample_rate):
        super().__init__()
        
        self.wave_lengths = wave_lengths
        self.wave_nos = 2. * torch.pi / wave_lengths
        self.distance = distance
        input_width = input_shape[2]
        input_heigth = input_shape[1]

        # coordinates of aperture
        self.x = torch.linspace(aperture_round[0], aperture_round[1], input_width).to(dtype=torch.complex64).cuda()
        self.y = torch.linspace(aperture_round[2], aperture_round[3], input_heigth).to(dtype=torch.complex64).cuda()

        # coordinates of sensor
        self.sx = torch.linspace(sensor_round[0], sensor_round[1], 88).to(dtype=torch.complex64).cuda() #88
        self.sy = torch.linspace(sensor_round[2], sensor_round[3], 88).to(dtype=torch.complex64).cuda() #88

        # range of frequency
        fxmin = 1.0 / wave_lengths * spatial_frenq[0]
        fxmax = 1.0 / wave_lengths * spatial_frenq[1]
        fymin = 1.0 / wave_lengths * spatial_frenq[2]
        fymax = 1.0 / wave_lengths * spatial_frenq[3]

        # frequency sampling
        self.fx = torch.linspace(fxmin, fxmax, asm_sample_rate).to(dtype=torch.complex64).cuda()
        self.fy = torch.linspace(fymin, fymax, asm_sample_rate).to(dtype=torch.complex64).cuda()
        
    def propagate(self, input_field):
        
        input_field = input_field[0, :, :, 0]
        fxx, fyy = torch.meshgrid(self.fx, self.fy, indexing='xy')
        fxx, fyy = fxx.to(dtype=torch.complex64).cuda(), fyy.to(dtype=torch.complex64).cuda()
        H_func = torch.exp(1j * self.wave_nos * self.distance * torch.sqrt(1 - (fxx * self.wave_lengths)**2 - (fyy * self.wave_lengths)**2))

        decompose_field = mdft(input_field, self.x, self.y, self.fx, self.fy).squeeze(0)
        prop_field = decompose_field * H_func
        output_field = midft_novec(prop_field, self.sx, self.sy, self.fx, self.fy, self.wave_lengths)

        return output_field.unsqueeze(3)