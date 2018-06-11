# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
#from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys
sys.path.append('..')
from lenet import LeNet



def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
    ap.add_argument("-m", "--model", required=True,
        help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args


args = args_parse()

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 200
INIT_LR = 1e-2
BS = 100
CLASS_NUM = 10
norm_size = 32
# initialize the data and labels

def get_data(images_path):
    if not os.path.exists(images_path):
        raise ValueError('images_path is not exist.')

    images = []
    labels = []
    images_path = os.path.join(images_path,'*.jpg')
    count = 0
    for image_file in glob.glob(images_path):
        count +=1
        if count % 100 == 0:
            print('Load{} images .'.format(count))
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (norm_size, norm_size))
        label = int(image_file.split('_')[-1].split('.')[0])
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)

    (trainX, testX, trainY, testY) = train_test_split(images,
            labels, test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=CLASS_NUM)
    testY = to_categorical(testY, num_classes=CLASS_NUM)   
    return trainX,trainY,testX,testY
   # return images,labels



    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
            labels, test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=CLASS_NUM)
    testY = to_categorical(testY, num_classes=CLASS_NUM)   
    return trainX,trainY,testX,testY


    

def train(aug,trainX,trainY,testX,testY,args):
    # initialize the model
    print("[INFO] compiling model...")
    model = LeNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#    opt = Adam(lr=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Invoice classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])
    


#python train.py --dataset ../../invoice_all/train  --model invoice.model
if __name__=='__main__':
    args = args_parse()
    file_path = args["dataset"]
    trainX,trainY,testX,testY = get_data(file_path)
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    
    train(aug,trainX,trainY,testX,testY,args)