import random
#from math import exp    #X=num_inp*num_features,Y=num_inp*num_features ,weights=num_features *num_labels
import numpy as np

class Logistic_regression(object):
	def __init__(self,X,Y,X_test,Y_test,num_labels,alpha,epochs,batch_size,validation_split):
		self.alpha=alpha
		self.epochs=epochs
		self.num_labels=num_labels
		self.X=X
		self.Y=Y
		self.batch_size=batch_size
		self.validation_split=validation_split
		self.weights=np.zeros((len(X[0]),num_labels))   #train labels should be 1,2......n
		self.bias=np.zeros((1,num_labels))
		self.X_test=X_test
		self.Y_test=Y_test
	def prediction(self,inputs):#1/(1+e^-(w'x+b))
		exp=2.718281828459045
		prediction=np.zeros(self.num_labels)
		logits=np.add(np.matmul(inputs,self.weights),self.bias) #x*x[0]  x x[0]*num_labels =x*num_labels
		prediction=1/(1+exp**(-1*logits))
		#print(np.shape(prediction[0]))
		return prediction[0]	

	def convert_labels_to_vec(self,label):#convert label 1 to [1,0,0,0..........n times]
		#try:
		v=np.zeros(self.num_labels)
		v[int(label)]=1
		v= np.transpose(v)
		#print np.shape(v)
		return v
		#except:
		#print ("error in labelling of data")
	def train_validation_split(self):
		#try:
		val_size=int(len(self.X)*self.validation_split)
		X_val=self.X[0:val_size]
		Y_val=self.Y[0:val_size]
		X_train=self.X[val_size:]
		Y_train=self.Y[val_size:]
		return X_train, Y_train, X_val, Y_val
		#except:
		print "check the dimensions of your input data and labels"
	def get_batch_data(self):
		X_train, Y_train,_,_=self.train_validation_split()
		try:
			offset=random.randint(0,len(self.X)-self.batch_size)
			batch_data=X_train[offset:offset+self.batch_size]
			batch_labels=Y_train[offset:offset+self.batch_size]
			return batch_data, batch_labels
		except:
			print("check your batch size")
	def gradient_descent_optimizer(self):                                                               # 1/1+e^-x
		batch_data , batch_labels = self.get_batch_data()
		for i in range(len(batch_data)):
			output_vec=self.convert_labels_to_vec(batch_labels[i])
			pred=self.prediction(batch_data[i]) #(1,num_labels)
			#error=np.subtract(output_vec,pred)
			for k in range(len(self.weights)):#i=num_features
				for j in range(len(self.weights[k])):#j=num_labels
					self.weights[k][j]=self.weights[k][j]+self.alpha*(output_vec[j]-pred[j])*(1-pred[j])*pred[j]*batch_data[i][k]
			for j in range(self.num_labels):
				self.bias[0][j]=self.bias[0][j]+self.alpha*(output_vec[j]-pred[j])*(1-pred[j])*pred[j]				
		
	def get_accuracy(self,epoch,X,Y):
		count=0
		for i in range(len(X)):
			pred=self.prediction(X[i])
			Y_pred=np.argmax(pred)
			#print np.shape(Y_pred),np.shape(Y)
			if Y_pred==Y[i]:
				count=count+1
		if epoch%100==0:
			print ("Epoch: ",epoch," ","Accuracy %: ",float(count)/float(len(X))*100)

			#get the max index of prediction[0]
	def get_test_accuracy(self):
		count=0
		for i in range(len(self.X_test)):
			pred=np.argmax(self.prediction(self.X_test[i]))
			if pred==self.Y_test[i]:
				count=count+1
		print("Accuracy on test data %:", float(count)/float(len(self.X_test))*100)


	def fit(self):
		_,_,X_val,Y_val=self.train_validation_split()
		for epoch in range(1,self.epochs+1):
			self.gradient_descent_optimizer()
			self.get_accuracy(epoch,X_val,Y_val)
		self.get_test_accuracy()
	def classifiy(self,X):
		pred=self.prediction(X)
		return np.argmax(pred)