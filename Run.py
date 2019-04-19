"""

This script utilises the ground truth label's 2D bounding box to 
crop out the the points of interest and feed it into the model so that 
it can predict a 3D bounding box for each of the 2D detections

The script will plot the results of the 3D bounding box onto the image
and display them alongside the groundtruth image and it's 3D bounding box.
This is to help with qualitative assesment. 

Images to be evaluated should be placed in Kitti/validation/image_2 

FLAGS:

--hide-imgs
Hides Display of ground truth and bounding box

"""

import os
import cv2
import errno
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
from lib.DataUtils import *
from lib.Utils import *

from lib import Model, ClassAverages

def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    weight_list = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]

    if len(weight_list) == 0:
        print('We could not find any model weight to load, please train the model first!')
        exit()
    else:
        print ('Using previous model weights %s'%weight_list[-1])
        my_vgg = models.vgg19_bn(pretrained=True)
        model = Model.Model(features=my_vgg.features, bins=2)
        if use_cuda: 
            checkpoint = torch.load(weights_path + '/%s'%weight_list[-1])
        else: 
            checkpoint = torch.load(weights_path + '/%s'%weight_list[-1],map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # Load Test Images from validation folder
    dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/Kitti/validation/')
    all_images = dataset.all_objects()
    print ("Length of validation data",len(all_images))
    averages = ClassAverages.ClassAverages()

    all_images = dataset.all_objects()
    print ("Model is commencing predictions.....")
    for key in sorted(all_images.keys()):

        data = all_images[key]
        truth_img = data['Image']
        img = np.copy(truth_img)
        imgGT = np.copy(truth_img)
        objects = data['Objects']
        cam_to_img = data['Calib']
   
        for object in objects:

            label = object.label
            theta_ray = object.theta_ray
            input_img = object.img

            input_tensor = torch.zeros([1,3,224,224])
            input_tensor[0,:,:,:] = input_img
            input_tensor.cuda()

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

            location = plot_regressed_3d_bbox_2(img, truth_img, cam_to_img, label['Box_2D'], dim, alpha, theta_ray)
            locationGT = plot_regressed_3d_bbox_2(imgGT, truth_img, cam_to_img, label['Box_2D'], label['Dimensions'], label['Alpha'], theta_ray)

            # print('Estimated pose: %s'%location)
            # print('Truth pose: %s'%label['Location'])
            # print('-------------')
        
        if not FLAGS.hide_imgs:
            
            numpy_vertical = np.concatenate((truth_img,imgGT, img), axis=0)
            cv2.imshow('2D detection on top, 3D Ground Truth on middle , 3D prediction on bottom', numpy_vertical)
            cv2.waitKey(0)

    print ("Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--hide-imgs", action="store_true",
                        help="Hide display of visual results")

    FLAGS = parser.parse_args()

    main()