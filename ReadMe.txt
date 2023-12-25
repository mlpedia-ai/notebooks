SimpleClassificationNN.ipynb

This notebook implements a simple neural network for classification in PyTorch. First, we generate some synthetic data for two classes by sampling from 2D multivariate gaussian distributions with different means. Then we aggregate those samples into a single tensor, and randomly set aside 20 percent of the samples as a test set, which we visualize. Next we define a simple neural network with a single hidden layer of 100 units. Since our labels are 0 or 1, the last activation is a sigmoid. We set Mean Square Error as the loss, and Stochastic Gradient Descent as the optimizer. The training loop iterates over the dataset 1000 times. Finally, we compute the error on the test set, and display the predictions.


SimpleRegressionNN.ipynb

This notebook implements a simple neural network for regression in PyTorch. First, we generate some synthetic data from a sine function, then randomly set aside 10 percent of the samples as a test set, which we visualize. Next we define a simple neural network with a single hidden layer of 64 units. We use L1 as the loss function, and Adam as the optimizer. The training loop iterates over the dataset 100 times. Finally, we compute the mean absolute error on the test set, and display the predictions.