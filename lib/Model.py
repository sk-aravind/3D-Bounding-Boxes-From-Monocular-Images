import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, features=None, bins=2, w = 0.4):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features
        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) 
                )
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins),
      
                )
        self.dimension = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )

    def forward(self, x):
        x = self.features(x) # 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(x)
        dimension = self.dimension(x)
        return orientation, confidence, dimension

def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):

   
    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])
    return (-1 * torch.cos(theta_diff - estimated_theta_diff).mean()) + 1

def detect_using_model(object,model,dataset,device,averages):

    label = object.label
    theta_ray = object.theta_ray
    input_img = object.img

    input_tensor = torch.zeros([1,3,224,224])
    input_tensor[0,:,:,:] = input_img
    input_tensor = input_tensor.cuda()

    [orient, conf, dim] = model(input_tensor)
    orient = orient.cpu().data.numpy()[0, :, :]
    conf = conf.cpu().data.numpy()[0, :]
    dim = dim.cpu().data.numpy()[0, :]

    dim += averages.get_item(label['Class'])

    argmax = np.argmax(conf)
    orient = orient[argmax, :]
    cos = orient[0]
    sin = orient[1]
    alpha = np.arctan2(sin, cos)
    alpha += dataset.angle_bins[argmax]
    alpha -= np.pi

    return dim,alpha,theta_ray,conf,label

   