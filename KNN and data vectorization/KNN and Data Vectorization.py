

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import sklearn
import random
import math

# ===================================================


def process_data(data, labels):
	"""
	Preprocess a dataset of strings into vector representations.

    Parameters
    ----------
    	data: numpy array
    		An array of N strings.
    	labels: numpy array
    		An array of N integer labels.

    Returns
    -------
    train_X: numpy array
		Array with shape (N, D) of N inputs.
    train_Y:
    	Array with shape (N,) of N labels.
    val_X:
		Array with shape (M, D) of M inputs.
    val_Y:
    	Array with shape (M,) of M labels.
    test_X:
		Array with shape (M, D) of M inputs.
    test_Y:
    	Array with shape (M,) of M labels.
	"""
	
	# Split the dataset of string into train, validation, and test 
	# Use a 70/15/15 split
	# train_test_split shuffles the data before splitting it 
	# Stratify keeps the proportion of labels the same in each split

	# -- WRITE THE SPLITTING CODE HERE -- 
	input_train, input_rem, label_train, label_rem = train_test_split(data,labels,test_size=0.3,train_size=0.7,random_state = 40,stratify=labels)
	input_valid, input_test, label_valid, label_test = train_test_split(input_rem,label_rem,test_size=0.5,train_size=0.5,random_state = 40,stratify=label_rem)

	# Preprocess each dataset of strings into a dataset of feature vectors
	# using the CountVectorizer function. 
	# Note, fit the Vectorizer using the training set only, and then
	# 	# transform the validation and test sets.
	vectorizer = CountVectorizer()
	Trainsetmatrixinput = vectorizer.fit_transform(input_train)
	vectorizer.fit(input_train)
	Validatesetmatrixinput = vectorizer.transform(input_valid)
	Testsetmatrixinput = vectorizer.transform(input_test)

	# Return the training, validation, and test set inputs and labels

	#RETURN THE ARRAYS
	return Trainsetmatrixinput,label_train,Validatesetmatrixinput,label_valid,Testsetmatrixinput,label_test

def select_knn_model(train_X, val_X, train_Y, val_Y):
	"""
	Test k in {1, ..., 20} and return the a k-NN model
	fitted to the training set with the best validation loss.

    Parameters
    ----------
    	train_X: numpy array
    		Array with shape (N, D) of N inputs.
    	val_X: numpy array
    		Array with shape (M, D) of M inputs.
    	train_Y: numpy array
    		Array with shape (N,) of N labels.
    	val_Y: numpy array
    		Array with shape (M,) of M labels.

    Returns
    -------
    best_model : KNeighborsClassifier
    	The best k-NN classifier fit on the training data 
    	and selected according to validation loss.
  	best_k : int
    	The best k value according to validation loss.
	"""

	minvalidateloss = None

	for k in range(1,21):

		knnmodel = KNeighborsClassifier(n_neighbors=k)
		knnmodel.fit(train_X,train_Y)
		predictions = knnmodel.predict(val_X)

		assert len(predictions) == len(val_Y)

		error_sum = 0
		for i in range(len(predictions)):
			if predictions[i] != val_Y[i]:
				error_sum += 1

		validationloss = error_sum / len(predictions)

		if minvalidateloss is None:
			minvalidateloss = (validationloss, k, knnmodel)
		else:
			if validationloss < minvalidateloss[0]:
				minvalidateloss = (validationloss, k, knnmodel)


	return minvalidateloss[2], minvalidateloss[1]



# Set random seed
np.random.seed(3142021)
random.seed(3142021)

def load_data():
	# Load the data
	with open('./clean_fake.txt', 'r') as f:
		fake = [l.strip() for l in f.readlines()]
	with open('./clean_real.txt', 'r') as f:
		real = [l.strip() for l in f.readlines()]

	# Each element is a string, corresponding to a headline
	data = np.array(real + fake)
	labels = np.array([0]*len(real) + [1]*len(fake))
	return data, labels


def main():
	data, labels = load_data()
	train_X, train_Y, val_X, val_Y, test_X, test_Y = process_data(data, labels)
	best_model, best_k = select_knn_model(train_X, val_X, train_Y, val_Y)
	test_accuracy = best_model.score(test_X, test_Y)
	print("Selected K: {}".format(best_k))
	print("Test Acc: {}".format(test_accuracy))


if __name__ == '__main__':
	main()
