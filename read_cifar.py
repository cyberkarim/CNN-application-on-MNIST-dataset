import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def unpickle(path):
    
    with open(path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def read_cifar_batch(data_batch_path):

     data_labels = unpickle(data_batch_path)
     img_data = data_labels[b'data'] 
     data_labels = data_labels[b'labels'] 
     
     return img_data,data_labels
     



def read_cifar(cifat_batches_py_path):
    data_batch_files = os.listdir(cifat_batches_py_path)
    data = []
    labels = []
    i = 0
    for data_batch_file in data_batch_files:
      if i == 1:
          break
      if data_batch_file != "batches.meta" and data_batch_file != "readme.html":
          data_batch_path = cifat_batches_py_path + "/" + data_batch_file
          imgs_data,imgs_label = read_cifar_batch(data_batch_path)
          for img_data in imgs_data:
             data += [img_data]
          for img_label in imgs_label:
             labels += [img_label]
          i = i + 1
    return np.asarray(data),np.asarray(labels)

def split_dataset(data,labels,split):
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=split)
    return data_train, data_test, labels_train, labels_test




if __name__ == "__main__":

    data,labels = read_cifar("data/cifar-10-batches-py")
    data_train, data_test, labels_train, labels_test = split_dataset(data,labels,split=0.05)