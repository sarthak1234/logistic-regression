from numpy import genfromtxt
from model import Logistic_regression
import numpy as np


no_of_images=42000
dataset = np.ndarray((no_of_images,784), dtype=np.float32)
dataset = genfromtxt('train.csv', delimiter=',')
print ("dataset loaded")
dataset=(dataset-127.5)/255.0


labels = np.ndarray(no_of_images, dtype=np.int32)
data_labels=genfromtxt('train_labels.csv',delimiter=',')
print("data_labels loaded")
training_dataset=dataset[0:30000][:]
test_dataset=dataset[30000:42000][:]
#print(training_dataset.shape,validation_dataset.shape)
#print(training_dataset[0])



#dividing training labels and validation labels
training_labels=data_labels[0:30000]
test_labels=data_labels[30000:42000]
batch_size=128
num_epochs=200
logistic_regression= Logistic_regression(X=training_dataset,Y=training_labels,X_test=test_dataset,Y_test=test_labels,epochs=num_epochs,batch_size=batch_size,alpha=0.1,validation_split=0.2,num_labels=10)
logistic_regression.fit()

