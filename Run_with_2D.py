"""

This script utilises the a yolo network to detect pedestrians and cars 
from and images. The 2D detections are crop out and fed it into the model so that 
it can predict a 3D bounding box for each of the 2D detections

The script will plot the results of the 3D bounding box onto the image and display it
using cv2.show, press the space bar in order to move on to the next image

Images to be evaluated should be placed in Kitti/validation/image_2 

FLAGS:
--val-img-path
Please specify the path to the images you wish to evaluate. 
Path default is Kitti/validation/image_2/

--calb-path
Please specify the path containing camera calibration obtained from KITTI. 
Path default is Kitti/camera_cal/

--show-2D
Shows yolonet's 2D BoundingBox detections of in a seperate image alongside the 3D regressed boxes

"""


import os
import time
import cv2
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from lib.DataUtils import *
from lib.Utils import *
from lib import Model, ClassAverages
from yolo.yolo import cv_Yolo


def main():

    bins_no = 2

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    weight_list = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(weight_list) == 0:
        print('We could not find any model weight to load, please train the model first!')
        exit()
    else:
        print('Using model weights : %s'%weight_list[-1])
        my_vgg = models.vgg19_bn(pretrained=True)
        model = Model.Model(features=my_vgg.features, bins=bins_no).to(device)
        if use_cuda: 
            checkpoint = torch.load(weights_path + '/%s'%weight_list[-1])
        else: 
            checkpoint = torch.load(weights_path + '/%s'%weight_list[-1],map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # Load Yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(bins_no)

    image_dir = FLAGS.val_img_path
    cal_dir = FLAGS.calb_path

    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file instead of per image calibration
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"
    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'
   
    try:
        ids = [x.split('.')[0][-6:] for x in sorted(glob.glob(img_path+'/*.png'))]
    except:
        print("\nError: There are no images in %s"%img_path)
        exit()

    for id in ids:
        start_time = time.time()
        img_file = img_path + id + ".png"

        # Read in image and make copy
        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)
        
        # Run Detection on yolo
        detections = yolo.detect(yolo_img)

        # For each 2D Detection
        for detection in detections:

            if not averages.recognized_class(detection.detected_class):
                continue
            # To catch errors should there be an invalid 2D detection
            try:
                object = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                continue

            theta_ray = object.theta_ray
            input_img = object.img
            proj_matrix = object.proj_matrix
            box_2d = detection.box_2d
            detected_class = detection.detected_class

            input_tensor = torch.zeros([1,3,224,224]).to(device)
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            if FLAGS.show_2D:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            else:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)
            
            print('Estimated pose: %s'%location)

        if FLAGS.show_2D:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
        else:
            cv2.imshow('3D detections', img)

        print("\n")
        print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
        print('-------------')

        
        if cv2.waitKey(0) != 32: # space bar
            exit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--val-img-path", default="Kitti/validation/image_2/",
                        help="Please specify the path to the images you wish to evaluate on.")

    parser.add_argument("--calb-path", default="Kitti/camera_cal/",
                        help="Please specify the path containing camera calibration obtained from KITTI")

    parser.add_argument("--show-2D", action="store_true",
                        help="Shows the 2D BoundingBox detections of the object detection model on a separate image")

    FLAGS = parser.parse_args()

    main()
