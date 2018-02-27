

""" This module performs linear regression using gradient descent iteration method. """

import numpy as np
import sklearn
from scipy import linalg
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


class NaiveBayes:
  """ This class performs the linear regression using gradient descent algorithm """
  def __init__(self):
    self.train_data =  np.array([])
    self.train_target =  np.array([])
    self.test_data =  np.array([])
    self.test_target =  np.array([])
    self.train = GaussianNB()

  def readfile(self, filename):
    dataset = np.array([])
    # read the sample data from file
    with open(filename, "rb") as data:
      f_data = data.read().decode('utf-8').split("\n")  #f_data = file data that has array of sample data

    #populate sample_features and target matrices with appropriate data
    invalid_samples = 0    # variable to hold the invalid sample count
    sample_features = np.array([])
    for sample in f_data:
      try:
        sample = sample.replace('\"',"")
        sample_data = list(map(float, sample.split(',')))
        dataset = np.concatenate((dataset, sample_data[1:]), axis=0)
      except(ValueError):
        invalid_samples = invalid_samples+1

    dataset.shape = (len(f_data)-invalid_samples, int(dataset.size/(len(f_data)-invalid_samples)))
    sample_features = dataset[:, :-1]
    target = np.array(dataset[:, -1])
    del dataset
    sample_features = np.concatenate((np.ones((len(f_data)-invalid_samples, 1), dtype='float'), sample_features), axis=1)
    self.splitdata(sample_features, target)
  
  def fit_data(self):
    self.train.fit(self.train_data, self.train_target)

  def splitdata(self,sample_features, target):
    self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(sample_features, target, test_size=0.33)
  
  def predict(self,test_data):
    return self.train.predict(test_data)

  def findscore(self):
    score = (self.predict(self.test_data) == self.test_target)
    return sum(score)/len(score)
  
if __name__ == "__main__":
  nb = NaiveBayes()
  nb.readfile("naive_bayes_dataset.csv")
  nb.fit_data()
  print (nb.findscore())
 


