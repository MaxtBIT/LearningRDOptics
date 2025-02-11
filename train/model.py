import torch.nn as nn
import train.CodedModel as CodedModel
import utils

# set the entire structure of the model
class DROptics(nn.Module):
    def __init__(self, codednet, reconnet):

        super(DROptics, self).__init__()
        self.codednet = codednet
        self.reconnet = reconnet
        
    def forward(self, x, mode, training):

        measure_pic, gts, psfs = self.codednet(x, mode, training)
        Output_hsi = self.reconnet(measure_pic, psfs)

        return gts, Output_hsi

def prepare_model(opt):

    print("exp_name:    ", opt.exp_name)
    if "FoVFormer" in opt.exp_name:
        import train.ReconModel as ReconModel
    else:
        reconmodel = None
        raise Exception("model_not_find")

    codedmodel = CodedModel.CodedModel()
    reconmodel = ReconModel.ReconModel() 
    model = DROptics(codedmodel, reconmodel)     
    
    if opt.cuda:
        model = model.cuda()
    else:
        raise NotImplementedError

    return model