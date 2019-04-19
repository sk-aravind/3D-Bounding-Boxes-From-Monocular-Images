"""

This script utilises the ground truth label's 2D bounding box to 
crop out the the points of interest and feed it into the model so that 
it can predict a 3D bounding box for each of the 2D detections

The script will plot the results of the 3D bounding box onto the image
and display them alongside the groundtruth image and it's 3D bounding box.
This is to help with qualitative assesment. 

Images to be evaluated should be placed in eval/image_2 folder 

Eval Results for each file in the eval/image_2 folder will be saved to "eval/eval-results/"


FLAGS:
--show-single
Show 3D BoundingBox detections one at a time

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
from tqdm import tqdm

from lib import Model, ClassAverages

def main():

    exp_no = 34

    print ("Generating evaluation results for experiment No. ",exp_no)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/exp_' + str(exp_no) + '/'
    weight_list = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]


    # Create out folder for pred-labels and pred-imgs

    for x in range(len(weight_list)):
        check_and_make_dir('Kitti/results/validation/labels/exp_' + str(exp_no)  +"/epoch_%s/" % str(x+1))
    check_and_make_dir('Kitti/results/validation/pred_imgs/exp_' + str(exp_no) )

    if len(weight_list) == 0:
        print('We could not find any model weights to load, please train the model first!')
        exit()
    
    for model_weight in weight_list:
        epoch_no = model_weight.split(".")[0].split('_')[-1]
        print ("Evaluating for Epoch: ",epoch_no)

        print ('Loading model with %s'%model_weight)
        my_vgg = models.vgg19_bn(pretrained=True)
        model = Model.Model(features=my_vgg.features, bins=2)
        if use_cuda: 
            checkpoint = torch.load(weights_path + model_weight)
        else: 
            checkpoint = torch.load(weights_path + model_weight)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load Test Images from eval folder
        dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + 'Kitti/validation')
        all_images = dataset.all_objects()
        print ("Length of eval data",len(all_images))
        averages = ClassAverages.ClassAverages()

        all_images = dataset.all_objects()
        print ("Model is commencing predictions.....")
        for key in tqdm(sorted(all_images.keys())):

            data = all_images[key]
            truth_img = data['Image']
            img = np.copy(truth_img)
            imgGT = np.copy(truth_img)
            objects = data['Objects']
            cam_to_img = data['Calib']

            filename =  "Kitti/results/validation/labels/exp_" +str(exp_no) + '/epoch_' + str(epoch_no) + "/" +str(key)+".txt"
            check_and_make_dir(filename)   
            file = open(filename,"w")

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

                file.write( \
                    #  Class label
                    str(label['Class']) + " -1 -1 " + \
                    # Alpha
                    str(round(alpha,2)) + " " + \
                    # 2D Bounding box coordinates
                    str(label['Box_2D'][0][0]) + " " + str(label['Box_2D'][0][1]) + " " + \
                        str(label['Box_2D'][1][0]) + " " + str(label['Box_2D'][1][1]) + " " + \
                    # 3D Box Dimensions
                    str(' '.join(str(round(e,2)) for e in dim)) + " " + \
                    # 3D Box Location
                    str(' '.join(str(round(e,2)) for e in location)) + " 0.0 " + \
                    # Ry
                    str(round(theta_ray + alpha ,2)) + " " + \
                    # Confidence
                    str( round(max(softmax(conf)),2) ) + "\n" 
                )

                # print('Estimated pose: %s'%location)
                # print('Truth pose: %s'%label['Location'])
                # print('-------------')



            file.close()
            
            
            numpy_vertical = np.concatenate((truth_img,imgGT, img), axis=0)
            image_name = 'Kitti/results/validation/pred_imgs/exp_' + str(exp_no) + '/' + str(key) + "/epoch_" + epoch_no + '_' + str(key) + '.jpg'
            check_and_make_dir(image_name)    
            cv2.imwrite(image_name, numpy_vertical)

        print ("Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--show-single", action="store_true",
                        help="Show 3D BoundingBox detecions one at a time")

    parser.add_argument("--hide-imgs", action="store_true",
                        help="Hide display of visual results")

    FLAGS = parser.parse_args()

    main()