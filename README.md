# Happywhale - Whale and Dolphin Identification
  Attempt at development of model that will match individual whales and dolphins by unique natural characteristics


# Overview
  The mission of this kaggle challenge was to increase global understanding and conservation efforts of marine wildlife. It aimed to do so by creating algorithms with trained models that could match dolphins and whales in the wild to track their movements and populations. The task to do so, was to create a model that will be trained with identified images of fins and make a set of predictions of which species the unidentified fins belong to. The approach I used was an image keras image training model with several layers in an attempt to train the model to classify the fins.


# Data
Data:

Input: train_images jpegs

Input: test_images jpegs

Input: train.csv with image and species name and individual ids Output: submission csv*

Size: 62.06 GB 

# Preprocessing
After creating the data frames, all species names were changed to either include whale or dolphin
Dataframe for images and species only was created as well
Data Visualization

Histogram of number of identified fins grouped by species

Histogram of total number of identified whales and dolphins


# Problem Formulation
Input/output: training images, testing images, training data frame and image dataframe were the inputs to train the model on image generation and classification. The goal was to have a submission csv output with the predictions

Model:
Keras dense layer training model that was generated to process images, such a model could be run multiple times in hopes of improving the model’s training and number of layers could be edited to manipulate results
# Training
Training was mainly done with software in a jupyter notebook. The model was simply compiled and it was trained with the images through about 2,400,000 parameters with categorical cross entropy as the loss metric.
Training took okay, hardware and space was a factor causing even small number of epochs to take a while

Loss vs epoch training and testing curve

Accuracy vs epoch training and testing curve
Major difficulty was downloading all the training images, they were twice as big as the testing images and my hardware could not handle that load so only about 60 images were used. This clearly caused the validity of this model to suffer since it had less to work with.


# Performance Comparison
Performance was based on classification, loss was of categorical cross entropy

# Conclusions 
Denser models with few epochs are better for image training
Splitting data frames with no validation generator was better for model but worse for classification
Alternate models could have been used like resnet
Hardware is a key factor with image training

# Future work
Use of multiple models to train with both images and ids
Future study could be a predictor challenge about an individual whale or dolphin’s swim path after matching and identification



# Citations
Functions used:
Path: https://docs.python.org/3/library/pathlib.html
Classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
Confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
Keras mobile net v2: https://keras.io/api/applications/mobilenet/#mobilenetv2-function
Tensorflow image data generator: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

# How to reproduce results
How to use model for other data:
Make a data frame for the images with a identifiable shape and certain amount of columns
Split that image dataframe into test and training 
Create generators for test and training
Compile pretrained model like mobilenet or resnet
Run layered model with the pretrained model as the input and softmax as the output
Use over 100 layers to properly process images and use less than 15 epochs for best results

# Files in repository
Test_images- folder of test images

Train_images- folder of sample of train images

Happywhale.ipynb- notebook with data manipulation, pretrained and keras model

Train.csv- training data of species

Sample_submission.csv- sample submission of predictions

# Software 
Numpy

Panda

Csv

Pathlib

Matplotlib

Os

Keras- layers, models, image processing and preprocessing, 
tensorflow

Sklearn- model, metrics

# Data
Where to download: https://www.kaggle.com/competitions/happy-whale-and-dolphin/data?select=test_images
