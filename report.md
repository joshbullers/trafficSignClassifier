#**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Data/WriteupImages/barChart.jpeg "Visualization"
[image2]: ./Data/WriteupImages/beforeProcessing.jpeg "Before Processing"
[image3]: ./Data/WriteupImages/afterProcessing.jpeg "After Processing"
[image4]: ./Data/OnlinePhotos/double-curve.resize.jpg "Traffic Sign 1"
[image5]: ./Data/OnlinePhotos/no-entry.resized.jpg "Traffic Sign 2"
[image6]: ./Data/OnlinePhotos/nopassing.resized.jpg "Traffic Sign 3"
[image7]: ./Data/OnlinePhotos/speed30.resized.jpg "Traffic Sign 4"
[image8]: ./Data/OnlinePhotos/stop-sign.resized.jpg "Traffic Sign 5"
[image9]: ./Data/WriteupImages/image1-prob.jpeg "Traffic Sign 1 Prob"
[image10]: ./Data/WriteupImages/image2-prob.png "Traffic Sign 2 Prob"
[image11]: ./Data/WriteupImages/image3-prob.png "Traffic Sign 3 Prob"
[image12]: ./Data/WriteupImages/image4-prob.png "Traffic Sign 4 Prob"
[image13]: ./Data/WriteupImages/image5-prob.png "Traffic Sign 5 Prob"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/joshbullers/trafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

A summary of the data is provided in the second code cell of the notebook.

I used .shape() to get information about the data and set() to get unique labels:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

This point is met by the code in the third code cell.

This visualization of the data shows the counts by class label.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

My preprocessing is fairly simple. The only processing applied is a histogram equalization. This helps brighten
the image or darken if too bright. Here is an example:

![alt text][image2]
![alt text][image3]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my architecture is in the seventh code cell. I have both LeNet and GGNet in there. I
used the GGNet architecture for this project.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 32x32x32    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32   				|
| RELU          		|           									|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64   				|
| RELU          		|           									|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x128     |
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 4x4x128					|
| Flatten				|												|
| Fully Connected		|												|
| RELU          		|												|
| Fully Connected		|												|
| RELU          		|												|
| Fully Connected		|												|
| Dropout       		|												|

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the notebook.

For training I used softmax cross entropy with logits and Adam Optimizer.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth and tenth cells of the notebook.

My final model results were:
* training set accuracy of 97.4%
* validation set accuracy of 94.6%
* test set accuracy of 93.7%

I first tried LeNet but quickly switched to this architecture defined above. I read about it in a
writeup and they referred to it as GGNet. GGNet is much deeper than LeNet and requires a small dropout
rate to avoid overfitting.

After a few attempts I moved the learning rate to .001 with 50 EPOCHS.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image will likely be difficult because the sign is further away. The other four are fairly standard
images and should be easy.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Double Curve     		| End of No Passing     						|
| No Entry     			| No Entry										|
| No Passing			| No Passing									|
| Speed Limit 30   		| Speed Limit 30				 				|
| Stop      			| Stop                							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is a little lower than the test set but still reasonable.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The probabilities were very close to 1 for images 2-5. Image 1 on the other hand the model was very uncertain.
The accurate prediction did not make the top 5.

![alt text][image9]
![alt text][image10]
![alt text][image11]  
![alt text][image12]
![alt text][image13]


