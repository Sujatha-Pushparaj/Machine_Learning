""" This module performs linear regression using gradient descent iteration method. """

import numpy as np
from scipy import linalg

class GradientDescent:
  """ This class performs the linear regression using gradient descent algorithm """
  def __init__(self):
    self.sample_features = np.array([])
    self.target = np.array([])
    self.parameter = np.array([])
    self.learning_rate = 5

  def train(self):
    """ This method performs gradient descendent and find the parameter matrix """
    self.parameter = np.ones(self.sample_features.shape[1])   
    for i in range (100000):
      for parameter_index in range(self.parameter.size):
        mean_sq_err = self.calc_mean_sq_err()
        tmp_parameter = self.parameter[parameter_index]
        self.parameter[parameter_index] = self.parameter[parameter_index] - (self.learning_rate*self.cost_func(parameter_index)) 
        if(self.calc_mean_sq_err() < mean_sq_err):
          self.learning_rate = self.learning_rate * 1.25
        else:
          self.learning_rate = self.learning_rate * 0.5
          self.parameter[parameter_index] = tmp_parameter
    print(self.parameter)
        
  def cost_func(self,parameter_index):
    estimated_target =[self.predict(sample) for sample in self.sample_features]
    cost=np.dot(estimated_target-self.target, self.sample_features[:, parameter_index])/self.sample_features.shape[0]
    return cost

  def calc_mean_sq_err(self):
    """ This method calculates the mean square error for the given parameter """
    estimated_target =[self.predict(sample) for sample in self.sample_features]
    #print (estimated_target, self.target)
    mean_sq_err = sum(list(x*x for x in (estimated_target-self.target)))
    return mean_sq_err
     
  def predict(self, feature):

    return np.dot(feature, self.parameter)
   

  def readfile(self, filename):
    dataset = np.array([])
    # read the sample data from file
    with open(filename, "rb") as data:
      f_data = data.read().decode('utf-8').split("\n")  #f_data = file data that has array of sample data

    #populate sample_features and target matrices with appropriate data
    invalid_samples = 0    # variable to hold the invalid sample count
    for sample in f_data:
      try:
        sample_data = list(map(float, sample.split(',')))
        dataset = np.concatenate((dataset, sample_data[1:]), axis=0)
      except(ValueError):
        invalid_samples = invalid_samples+1

    dataset.shape = (len(f_data)-invalid_samples, int(dataset.size/(len(f_data)-invalid_samples)))
    self.sample_features = dataset[:, :-1]
    self.target = np.array(dataset[:, -1])
    del dataset
    self.sample_features = np.concatenate((np.ones((len(f_data)-invalid_samples, 1), dtype='float'), self.sample_features), axis=1)


if __name__ == "__main__":
  grad_dec = GradientDescent()
  grad_dec.readfile("dataset1.csv")
  grad_dec.train()
  try:
    test_in = list(map(float, input("Enter the sample feature").split(" ")))
  except:
    test_in = []

  test_in.insert(0, 1.)
  rst=grad_dec.predict(test_in)
  print (rst)
  
