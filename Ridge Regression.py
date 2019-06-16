import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import copy

%matplotlib inline

## Ridgre Regression to calculate weights
def mylinridgereg(X, Y, lamda):
	n_row = (X.T).shape[0]
	Iden = np.identity(n_row)
	first_term = np.linalg.inv((np.matmul((X.T),X)) + (lamda*Iden))
	weight_mat = np.matmul(np.matmul(first_term,(X.T)),Y)
	return weight_mat
	
def mylinridgeregeval(X, weights):
	y_pred = np.matmul(X,weights)
	return y_pred
	
def meansquarederr(T, Tdash):
	mse = np.square(np.subtract(T, Tdash)).mean()
	return mse
	
# Load data set
with open("linregdata") as f:
    data = [line for line in csv.reader(f, delimiter=",")]
ip_size = len(data)
print("Number of records in input file: %d" % ip_size)

## Extract gender data from input data
gender_data = [i[0] for i in data]

## One hot encoding of gender data as (M,F,I)
gender_onehot = []
for ix in gender_data:
	if ix == 'M':
		gender_onehot.append([1,0,0])
	elif ix == 'F':
		gender_onehot.append([0,1,0])
	else:
		gender_onehot.append([0,0,1])

## Delete the existing gender element from list		
for ix in data:
	del ix[0]

## Append one hot encoded gender list to input data
for ix in range(len(data)):
	data[ix] =  gender_onehot[ix] + data[ix]
	
ipdata = copy.deepcopy(np.mat(data).astype(np.float))

## Split data into train and test
train_set,test_set = train_test_split(ipdata,test_size=0.20,random_state=5)

Xtrain_mat = np.c_[ train_set[:,:-1], np.ones(train_set.shape[0]) ]

train_mean_vec = Xtrain_mat[:,:-1].mean(axis=0)
train_std_vec = Xtrain_mat[:,:-1].std(axis=0)

## Standardize training data using training mean and standard deviation
Xtrain_mat[:,:-1] = (Xtrain_mat[:,:-1]-train_mean_vec)/train_std_vec

## Calculate weights using Ridge regression with lambda = 0.1	
weight_mat = mylinridgereg(Xtrain_mat,train_set[:,-1],24.6)

## Predict  using the weights derived from Ridge regression (above) on training set
y_train_pred = mylinridgeregeval(Xtrain_mat,weight_mat)

## Compute mean square error of predicted and actual target values of training data
mse_train = meansquarederr(train_set[:,-1], y_train_pred)

Xtest_mat = np.c_[ test_set[:,:-1], np.ones(test_set.shape[0]) ]

## Standardize test data using training mean and standard deviation
Xtest_mat[:,:-1] = (Xtest_mat[:,:-1]-train_mean_vec)/train_std_vec

## Predict target variable on test set
y_test_pred = mylinridgeregeval(Xtest_mat,weight_mat)

## Compute mean square error of predicted and actual target values of test data
mse_test = meansquarederr(test_set[:,-1], y_test_pred)

plt.plot(y_train_pred, train_set[:,-1], 'x',color='red')
plt.xlabel("Predicted Target value")
plt.ylabel("Actual Target value")
plt.title("Predicted vs Target Target values for Training set")
plt.show()

plt.plot(y_test_pred, test_set[:,-1], 'o',color='red')
plt.xlabel("Predicted Target value")
plt.ylabel("Actual Target value")
plt.title("Predicted vs Target Target values for Test set")
plt.show()