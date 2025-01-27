import torch

def mdft(in_matrix, x, y, fx, fy):

    x = x.unsqueeze(-1)
    y = y.unsqueeze(-2)
    fx = fx.unsqueeze(-2).unsqueeze(0)
    fy = fy.unsqueeze(-1).unsqueeze(0)
    mx = torch.exp(-2 * torch.pi * 1j * torch.matmul(x, fx))
    my = torch.exp(-2 * torch.pi * 1j * torch.matmul(fy, y))
    out_matrix = torch.matmul(torch.matmul(my, in_matrix), mx)

    lx = torch.numel(x)
    ly = torch.numel(y)
    if lx == 1:
        dx = 1
    else:
        dx = (torch.squeeze(x)[-1] - torch.squeeze(x)[0]) / (lx - 1)

    if ly == 1:
        dy = 1
    else:
        dy = (torch.squeeze(y)[-1] - torch.squeeze(y)[0]) / (ly - 1)

    out_matrix = out_matrix * dx * dy

    return out_matrix

def midft(in_matrix, x, y, fx, fy, wave_lengths):

    fxfx, fyfy = torch.meshgrid(fx, fy, indexing='xy')
    
    x = x.unsqueeze(-2)
    y = y.unsqueeze(-1)
    fx = fx.unsqueeze(-1).unsqueeze(0)
    fy = fy.unsqueeze(-2).unsqueeze(0)

    mx = torch.exp(2 * torch.pi * 1j * torch.matmul(fx, x))
    my = torch.exp(2 * torch.pi * 1j * torch.matmul(y, fy))
    out_matrix = torch.matmul(torch.matmul(my, in_matrix), mx)
    
    with torch.no_grad():
        out_matrix_x = torch.matmul(torch.matmul(my, (in_matrix * fxfx * wave_lengths)), mx)
        out_matrix_y = torch.matmul(torch.matmul(my, (in_matrix * fyfy * wave_lengths)), mx)
        out_matrix_z = torch.matmul(torch.matmul(my, (in_matrix * torch.sqrt((1.0 - (fxfx * wave_lengths)**2 - (fyfy * wave_lengths)**2)))), mx)
        out_matrix_sum = torch.sqrt(torch.abs(out_matrix_x)**2 + torch.abs(out_matrix_y)**2 + torch.abs(out_matrix_z)**2)

        angle1 = torch.atan2(out_matrix_x.imag, out_matrix_x.real)
        angle2 = torch.atan2(out_matrix_y.imag, out_matrix_y.real)
        angle3 = torch.atan2(out_matrix_z.imag, out_matrix_z.real)

        cos_x = torch.abs(out_matrix_x) / out_matrix_sum * torch.sign(angle3 * angle1)
        cos_y = torch.abs(out_matrix_y) / out_matrix_sum * torch.sign(angle3 * angle2)
        cos_z = torch.abs(out_matrix_z) / out_matrix_sum

        ray_vector = torch.cat((cos_x.unsqueeze(3), cos_y.unsqueeze(3), cos_z.unsqueeze(3)), dim=3)
        ray_vector = ray_vector.to(dtype=torch.float32).squeeze(0).unsqueeze(3)

    lfx = torch.numel(fx)
    lfy = torch.numel(fy)
    if lfx == 1:
        dfx = 1
    else:
        dfx = (torch.squeeze(fx)[-1] - torch.squeeze(fx)[0]) / (lfx - 1)

    if lfy == 1:
        dfy = 1
    else:
        dfy = (torch.squeeze(fy)[-1] - torch.squeeze(fy)[0]) / (lfy - 1)

    out_matrix = out_matrix * dfx * dfy

    return out_matrix, ray_vector

def midft_novec(in_matrix, x, y, fx, fy, wave_lengths):

    x = x.unsqueeze(-2)
    y = y.unsqueeze(-1)
    fx = fx.unsqueeze(-1).unsqueeze(0)
    fy = fy.unsqueeze(-2).unsqueeze(0)
    
    mx = torch.exp(2 * torch.pi * 1j * torch.matmul(fx, x))
    my = torch.exp(2 * torch.pi * 1j * torch.matmul(y, fy))
    out_matrix = torch.matmul(torch.matmul(my, in_matrix), mx)

    lfx = torch.numel(fx)
    lfy = torch.numel(fy)
    if lfx == 1:
        dfx = 1
    else:
        dfx = (torch.squeeze(fx)[-1] - torch.squeeze(fx)[0]) / (lfx - 1)

    if lfy == 1:
        dfy = 1
    else:
        dfy = (torch.squeeze(fy)[-1] - torch.squeeze(fy)[0]) / (lfy - 1)

    out_matrix = out_matrix * dfx * dfy
    return out_matrix

def complex_exponent_tf(phase):
    real = torch.cos(phase)
    imag = torch.sin(phase)
    z = torch.complex(real, imag)
    return z.to(torch.complex64)