import numpy as np
from scipy import linalg

import sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)

X=np.array([])
Y=np.array([])

# read the sample data from file
with open("dataset1.csv", "rb") as data:
  f_data=data.read().decode('utf-8').split("\n")  #f_data = array of each sample data

#populate X and Y matrices with appropriate data
str_inv=0    #variable to hold the invalid sample count
for sample in f_data:
  try:
    x=list(map(float,sample.split(',')))
    X=np.concatenate((X,x[1:]),axis=0)
  except(ValueError):
    str_inv=str_inv+1

X.shape=(len(f_data)-str_inv , int(X.size/(len(f_data)-str_inv)))
Y=X[:,-1]
X=X[:,:-1]
#X=np.concatenate((np.ones((50,1),dtype='float'),X),axis=1)
model.fit(X, Y)
print("coeff:",model.coef_,"inter:",model.intercept_)

try:
  test_in=list(map(float,input("Enter the sample feature").split(" ")))
except:
  pass
#test_in.insert(0,1.)

print(model.predict(*test_in))
