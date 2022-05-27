# Yelp Restaurant Photo Classification
Code for the Yelp Kaggle contest: https://www.kaggle.com/c/yelp-restaurant-photo-classification

My final submission which secured 16th place is a slightly enhanced version of this

![image](https://user-images.githubusercontent.com/9631296/170640454-1a48cc07-5255-498a-a405-2a3b643c5682.png)



## Approach
- Obtain feature vectors for each image from the pretrained MXNET model
- Build XGBoost models from these feature vectors
- Stack the multiple XGBoost models for the final prediction

**Stack_NN**: The above two models should give us 5 files (1 from Model 1 and 4 from Model2). These files are ensembled using a Neural Network.

This is a multi label classification problem. However, since XGBoost does not support multi label classification out of the box, Models 1 and 2 were built as binary classfication problems. This deficiency which does not take the dependancy between the labels into account, was taken care of by the ensembling stage.

The final ensemble was built using a Neural network, hence it was treated as a multilabel classification problem by having 9 nodes in the output layer.
