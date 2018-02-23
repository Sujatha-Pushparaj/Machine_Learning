""" This module performs linear regression using normal matrix equation method """ 

import numpy as np
from scipy import linalg

#Initialise two numpy arrays 
dataset = np.array([])

# read the sample data from file
with open("dataset1.csv", "rb") as data:
  f_data = data.read().decode('utf-8').split("\n")  #f_data = file data that has array of sample data

#populate sample_features and target matrices with appropriate data
invalid_samples = 0    # variable to hold the invalid sample count
for sample in f_data:
  try:
    sample_data = list(map(float,sample.split(',')))
    dataset = np.concatenate((dataset, sample_data[1:]), axis=0)
  except(ValueError):
    invalid_samples = invalid_samples+1

dataset.shape = (len(f_data)-invalid_samples, int(dataset.size/(len(f_data)-invalid_samples)))
sample_features = dataset[:,:-1]
target = np.array(dataset[:,-1])
del dataset
sample_features = np.concatenate((np.ones((len(f_data)-invalid_samples, 1), dtype='float'), sample_features), axis=1)

#calculate the hypothesis

parameter = np.matmul(np.matmul(linalg.inv(np.matmul(np.transpose(sample_features), sample_features)), np.transpose(sample_features)), target)
#parameter=np.matmul(np.linalg.pinv(sample_features),target)
print (parameter)

#test with user data
try:
  test_in = list(map(float, input("Enter the sample feature").split(" ")))
except:
  test_in = []

test_in.insert(0, 1.)
print(np.matmul(test_in, np.transpose(parameter)))





  
