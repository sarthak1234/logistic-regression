# logistic-regression
A simple numpy based implementation of logistic regression, gives upto 90% accuracy with MNIST hand written digits  data.
model.py contains the logistic model which is called to train the MNIST data in classifier.py.
Stochastic gradient descent has been used to optimize the logistic function.
The model follows batched method of training.
X,Y in the model are training data with
alpha=learning rate and
validation_split= ratio of training data to be split to validate the training.
for more info regarding logistic regression algorithm refer to http://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning
