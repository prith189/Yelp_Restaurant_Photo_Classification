# Yelp Restaurant Photo Classification
Code for the Yelp Kaggle contest: https://www.kaggle.com/c/yelp-restaurant-photo-classification

Thanks to Nina Chen for providing the starter code: https://www.kaggle.com/c/yelp-restaurant-photo-classification/forums/t/19206/deep-learning-starter-code

And to the MXNET team for providing the pretrained Inception net: https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md

My final submission which secured 16th place is a slightly enhanced version of this


##Approach

In order to obtain the features for each of the photographs, we use the pretrained inception model and obtain the values of the final fully connected layer for each image.

**Model 1**: Build a feature set for each business by taking the mean of features of all the images that belong to the business. Build 9 binary classification models (one for each label) using XGBoost. Use the results from this in the final ensemble.

**Model 2**: Build a feature set for each business by first clustering similar images (cluster sizes: 2,3,4,5), and then taking the mean of the features of similar image clusters. Build 9 binary classification models (one for each label) using XGBoost. Use the results from this in the final ensemble (This step will output four different models, one for each cluster size)

**Stack_NN**: The above two models should give us 5 files (1 from Model 1 and 4 from Model2). These files are ensembled using a Neural Network.

This is a multi label classification problem. However, since XGBoost does not support multi label classification out of the box, Models 1 and 2 were built as binary classfication problems. This deficiency which does not take the dependancy between the labels into account was taken care of by the ensembling stage.

The ensemble was built using a Neural network, hence it was treated as a multilabel classification problem by having 9 nodes in the output layer.

##Running the model

1. Download the data from Kaggle and unzip train_photos and test_photos

2. Run the pretraining script to generate the features from images. This will generate two numpy files 'feat_holder.npy' containing the features for all training photos, and similarly 'feat_holder_test.npy'

3. Generate train_cl.csv by using the clean_train function in the file Model1.py

4. Run Model1.py. This will generate two files 'Model1_Full.npy' and 'Model1_Full_result.npy'

5. Run Model2.py. This will generate eight files 'Model_Full_2.npy', 'Model1_Full_2_result.npy' as so on for each of the different cluster sizes

6. Run Stack_NN.py. This script should generate the final ensembled result.


