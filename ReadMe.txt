SimpleClassificationNN.ipynb

This notebook implements a simple neural network for classification in PyTorch. First, we generate some synthetic data for two classes by sampling from 2D multivariate gaussian distributions with different means. Then we aggregate those samples into a single tensor, and randomly set aside 20 percent of the samples as a test set, which we visualize. Next we define a simple neural network with a single hidden layer of 100 units. Since our labels are 0 or 1, the last activation is a sigmoid. We set Mean Square Error as the loss, and Stochastic Gradient Descent as the optimizer. The training loop iterates over the dataset 1000 times. Finally, we compute the error on the test set, and display the predictions.


SimpleRegressionNN.ipynb

This notebook implements a simple neural network for regression in PyTorch. First, we generate some synthetic data from a sine function, then randomly set aside 10 percent of the samples as a test set, which we visualize. Next we define a simple neural network with a single hidden layer of 64 units. We use L1 as the loss function, and Adam as the optimizer. The training loop iterates over the dataset 100 times. Finally, we compute the mean absolute error on the test set, and display the predictions.


DigitClassificationCNN.ipynb

This notebook implements a Convolutional Neural Network to classify images with written digits. First we define a function to create images with digits with some randomness. We then create a dataset with 1000 training images and 100 test images per class. In this example we use the Dataset and DataLoader classes from PyTorch to manage reading data and creating batches. Our CNN has two convolution layers, each followed by ReLU and pooling, a flattening layer, followed by dropout, and three fully connected layers, two of which are followed by ReLU activations. We use the Cross Entropy Loss, which is typical in classification tasks. After training, we measure the accuracy on the test set, and display some predictions on newly generated digits.


SiameseCNN.ipynb

This notebook implements a Siamese Convolutional Neural Network to predict if two given images correspond to the same digit or not. First we define a function to create images with digits from 0 to 4 with some randomness. We then create training and test samples for each digit. Following that we define a function to get batches for training and testing; half of the batch contains image pairs of the same digit. We then define the branch of the Siamese CNN, which maps the image input into a vector of dimension 64, and the siamese model, which merges the two branches using absolute difference, and outputs one of two classes: 'same' or 'different'. Then we train the model as a 2-class classification problem. We show the training curve, the accuracy on 100 batches from the test set, and some predictions.