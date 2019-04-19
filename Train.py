"""

This script trains the model using cropped images of cars and pedestrians 
obtained from the labels. The model produces 3 outputs the orientation,dimension 
and it's confidence of the 3D bounding box based on the 2D image. 

For each epoch we save the evaluation results separately to perform qualitative
and quantitative offline evaluation later. 
Quantitative results are recorded in the same format as kitti label data. For more indepth
offline evaluation after training 
Qualitative results such as how the model is predicting the 3D bounding boxes is also saved 
every epoch so we can see how the model is evolving and improving over time. 

"""

import os
import torch
import cv2
import argparse
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchsummary import summary
from tqdm import tqdm
from lib.DataUtils import *
from lib.Utils import *
from lib.Model import *
from lib import ClassAverages


from torch.utils import data as torch_data
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def main():
    print ("Initializing....")
    # ======= Hyper Parameters ======== #
    epochs = FLAGS.epochs
    batch_size = 16
    lr = 0.0001
    momentum = 0.9

    alpha = 0.6 #dimen
    w = 0.7 # orient

    exp_no = FLAGS.exp_no

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 6}
    # ================================== #

    print ("Starting Experiment No. ",exp_no)
    print ("Training for {} epochs ".format(epochs))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print ("Pytorch is using : ",device)

    print("Loading data...")
    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path)

    generator = torch_data.DataLoader(dataset, **params)
    eval_dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/eval/train')
    averages = ClassAverages.ClassAverages()

    print("Loading model...")
    my_vgg = models.vgg19_bn(pretrained=True)
    model = Model(features=my_vgg.features).cuda()

    ## SHOW SUMMARY OF MODEL 
    # summary(model,(3,244,244))

    opt_SGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss

    # load any previous weights
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/'
    latest_model = None
    first_epoch = 0
    
    
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass

    # Create Folders for this experiments
    for x in range(epochs):
        check_and_make_dir('Kitti/results/training/plots/exp_'+ str(exp_no) +"/epoch_%s/" % str(x+1))
    check_and_make_dir(weights_path + "exp_"+str(exp_no) +'/')

    if latest_model is not None:
        
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        # else: 
        #     checkpoint = torch.load(weights_path + '/%s'%model_lst[-1],map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print('Found previous checkpoint: %s at epoch %s'%(latest_model, first_epoch))
        print('Resuming training....')


    total_num_batches = int(len(dataset) / batch_size)
    losses=[]
    epoch_losses=[]
    dim_lossess=[]
    theta_lossess=[]
    orient_lossess=[]
    
    print('Training is commencing....')
    for epoch in range(first_epoch+1, epochs+1):
        # model.train(True)
        curr_batch = 0
        passes = 0
        # Training Loop
        for local_batch, local_labels in generator:

            truth_orient = local_labels['Orientation'].float().cuda()
            truth_conf = local_labels['Confidence'].long().cuda()
            truth_dim = local_labels['Dimensions'].float().cuda()

            local_batch=local_batch.float().cuda()
            [orient, conf, dim] = model(local_batch)

            orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
            dim_loss = dim_loss_func(dim, truth_dim)

            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)

            loss_theta = conf_loss + w * orient_loss
            loss = alpha * dim_loss + loss_theta

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()


            if passes % 50 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %s]" %(epoch, curr_batch, total_num_batches, loss.item()))
                passes = 0

            orient_lossess.append(orient_loss.item())
            dim_lossess.append(dim_loss.item())
            theta_lossess.append(loss_theta.item())
            losses.append(loss.item())
            passes += 1
            curr_batch += 1

       
        epoch_losses.append(loss.item())
        ### ++++++++++++++++++++++++++++++++++++++++++++
        # save after every 10 epochs
        if epoch % 1 == 0:
            name = weights_path + "exp_"+ str(exp_no) + "/exp_"+ str(exp_no) + '_epoch_%s.pkl' % epoch
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss
                    }, name)
            print("====================")


        path = 'Kitti/results/training/plots/exp_'+ str(exp_no) + '/'
        print ("Saving Metric Graphs")



        plt.figure(figsize=(20,8))
        plt.plot(orient_lossess)
        plt.ylabel('Overall Loss')
        plt.xlabel('Iterations')
        plt.savefig( path + "epoch_%s/" % epoch + "exp_"+str(exp_no) +"_epoch_%s" % epoch + "_Orientation.png")
        plt.clf()

        plt.figure(figsize=(20,8))
        plt.plot(dim_lossess)
        plt.ylabel('Dimension Loss')
        plt.xlabel('Iterations')
        plt.savefig( path + "epoch_%s/" % epoch + "exp_"+str(exp_no) +"_epoch_%s" % epoch + '_Dimension.png')
        plt.clf()

        plt.figure(figsize=(20,8))
        plt.plot(theta_lossess)
        plt.ylabel('Theta Loss')
        plt.xlabel('Iterations')
        plt.savefig( path + "epoch_%s/" % epoch + "exp_"+str(exp_no) +"_epoch_%s" % epoch + '_Theta.png')
        plt.clf()

        plt.figure(figsize=(20,8))
        plt.plot(losses)
        plt.ylabel('Overall Loss')
        plt.xlabel('Iterations')
        plt.savefig( path + "epoch_%s/" % epoch + "exp_"+str(exp_no) +"_epoch_%s" % epoch + '_Overall-Loss.png')
        plt.clf()

        plt.figure(figsize=(20,8))
        plt.plot(epoch_losses)
        plt.ylabel('Overall Loss')
        plt.xlabel('Epoch')
        plt.savefig( path + "epoch_%s/" % epoch + "exp_"+str(exp_no) +"_epoch_%s" % epoch + '_Overall-Loss-per-Epoch.png')
        plt.clf()

    result_name = 'Kitti/results/training/results_exp_' +str(exp_no)+".txt"

    check_and_make_dir(result_name)


    file = open(result_name,"w")

    best_loss_epoch_index = np.argmin(epoch_losses) 
    best_loss = epoch_losses[best_loss_epoch_index]

    file.write( "Epoch with the lowest loss : Epoch " + str(best_loss_epoch_index) + "   Loss: " + str(best_loss) + "\n" )

    best_loss_epoch_index = np.argmin(orient_lossess) 
    best_loss = orient_lossess[best_loss_epoch_index]

    file.write( "Iteration with the lowest orientation loss : Iteration " + str(best_loss_epoch_index) + "   Loss: " + str(best_loss) + "\n" )

    best_loss_epoch_index = np.argmin(dim_lossess) 
    best_loss = dim_lossess[best_loss_epoch_index]

    file.write( "Iteration with the lowest dimension loss : Iteration " + str(best_loss_epoch_index) + "   Loss: " + str(best_loss) + "\n" )

    best_loss_epoch_index = np.argmin(theta_lossess) 
    best_loss = theta_lossess[best_loss_epoch_index]

    file.write( "Iteration with the lowest theta loss : Iteration " + str(best_loss_epoch_index) + "   Loss: " + str(best_loss) + "\n" )
   
    file.close()

if __name__=='__main__':

    

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=2,
                    help="No of training epochs")

    parser.add_argument("--exp-no", type=int, required=True,
                help="Experiment No. so we can save all the metrics and weights related to this experiment")


    FLAGS = parser.parse_args()
    
    main()
