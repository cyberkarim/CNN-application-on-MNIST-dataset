import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt 
from read_cifar import read_cifar, split_dataset


def sigma(z2):
    return 1 / (1 + np.exp(-z2)) 


def sigma_prime(z2):
    return sigma(z2)*(1-sigma(z2))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def binary_cross_entropy(yhat: np.ndarray, y: np.ndarray) -> float:
    """Compute binary cross-entropy loss for a vector of predictions

    Parameters
    ----------
    yhat
        An array with len(yhat) predictions between [0, 1]
    y
        An array with len(y) labels where each is one of {0, 1}
    """
    return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean()

#def learn_once_mse(w1,b1,w2,b2,data,targets,learning_rate)
def learn_once_mse():
  N = 30  # number of input data
  d_in = 3  # input dimension
  d_h = 3  # number of neurons in the hidden layer
  d_out = 2  # output dimension (number of neurons of the output layer)

# Random initialization of the network weights and biaises
  w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
  b1 = np.zeros((1, d_h))  # first layer biaises
  w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
  b2 = np.zeros((1, d_out))  # second layer biaises
  b3 = np.zeros((1, d_out))  #
  data = np.random.rand(N, d_in)  # create a random data
  targets = np.random.rand(N, d_out)  # create a random targets

# Forward pass
  a0 = data # the data are the input of the first layer
  z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
  a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
  z2 = np.matmul(a1, w2) + b2  # input of the output layer
  a2 = 1 / (1 + np.exp(-z2))  # output of the output layer (sigmoid activation function)
  predictions = a2  # the predicted values are the outputs of the output layer

# Compute loss (MSE)
  loss = np.mean(np.square(predictions - targets))

  dLdW2 = np.matmul(np.transpose(a2),(2/d_out)*(a2-targets)*sigma_prime(a2))
  dLdb2 = (2/d_out)*(a2-targets)*sigma_prime(a2)
  dLdW1 = np.matmul(np.transpose(a2),np.matmul((2/d_out)*(a2-targets),np.transpose(w2)*sigma_prime(a1)))
  dLdb1 = np.matmul((2/d_out)*(a2-targets),np.transpose(w2)*sigma_prime(a1))
  #print(np.transpose(w2)*sigma_prime(a2))
  return dLdW2


#def learn_once_cross_entropy(w1,b1,w2,b2,data,targets,learning_rate):
def Learn_once_cross_entropy(learning_rate):
  N = 30  # number of input data
  d_in = 3  # input dimension
  d_h = 64  # number of neurons in the hidden layer
  d_out = 2  # output dimension (number of neurons of the output layer)

  # Random initialization of the network weights and biaises
  w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
  b1 = np.zeros((1, d_h))  # first layer biaises
  w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
  b2 = np.zeros((1, d_out))  # second layer biaises

  data = np.random.rand(N, d_in)  # create a random data
  targets = np.random.rand(N, d_out)  # create a random targets

  # Forward pass
  a0 = data # the data are the input of the first layer
  z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
  a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
  z2 = np.matmul(a1, w2) + b2  # input of the output layer
  a2 = softmax(z2)  # output of the output layer (BINARY CROSS ENTROPY activation function)
  predictions = a2  # the predicted values are the outputs of the output layer
  print(a2.shape)

  # Compute loss (MSE)
  loss = binary_cross_entropy(predictions,targets)

  #dLdW2 = np.matmul(np.transpose(a2),(a2-targets))
  dLdW2 = np.matmul(np.transpose(a2),a2-targets)
  dLdb2 = a2-targets
  #dLdb2 = np.zeros((1, d_out))  # first layer biaises
  #dLdW1 = 2 * np.random.rand(d_in, d_h) - 1
  return w2 + learning_rate*dLdW2
# b2 + learning_rate*dLdb2

def learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate):
    Z1 = np.dot(w1.T, data.T).T + b1
    A1 = sigma(Z1)
    Z2 = np.dot(w2.T, A1.T).T + b2
    A2 = softmax(Z2)
    
    Y = labels_train
    Y = Y.T
    
    delta2 = A2.T - Y
    dL_dw2 = np.dot(delta2, A1) / data.shape[0]
    
    dL_db2 = np.sum(delta2, axis=1, keepdims=True).T/ data.shape[0]
    #print(dL_db2.shape)
    dA1 = np.dot(w2, delta2)

    
    #print(dA1.T.shape)

    dZ1 = dA1.T * A1* (1 - A1)
    delta1 = dA1.T * A1 * (1 - A1)
    dL_dw1 = np.dot(dZ1.T, data) / data.shape[0]
    dL_db1 = np.sum(delta1, axis=0, keepdims=True)/ data.shape[0]

    w1 -= learning_rate * dL_dw1.T
    b1 -= learning_rate * dL_db1
    w2 -= learning_rate * dL_dw2.T
    b2 -= learning_rate * dL_db2

    
    loss = -np.mean(-Y * np.log(A2.T))

    return w1, b1, w2, b2, loss
# b2 + learning_rate*dLdb2

  


def one_hot(labels):

    labels_set = set(labels)

    one_hot_encoding = [[0]*len(labels_set)]*len(labels)

    for i in range(0,len(one_hot_encoding)):
        one_hot_encoding[i][labels[i]] = 1

    return np.array(one_hot_encoding)




def train_mlp(w1,b1,w2,b2,data_train,labels_train,learning_rate,num_epoch):
  
    one_hot_labels = one_hot(labels_train)
    accuracies = []
    for i in range(0,num_epoch):
       w1, b1, w2, b2, loss = learn_once_cross_entropy(w1,b1,w2,b2,data_train,one_hot_labels,learning_rate)
       a0 = data_train # the data are the input of the first layer
       z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
       a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
       z2 = np.matmul(a1, w2) + b2  # input of the output layer
       a2 = softmax(z2) # output of the output layer (BINARY CROSS ENTROPY activation function)
       predictions = a2  # the predicted values are the outputs of the output layer
       accuracy = 0
       #for prediction,label in (predictions,one_hot_labels):
       for i in range(0,len(predictions)):
        if list(predictions[i]).index(max(predictions[i])) == list(one_hot_labels[i]).index(max(one_hot_labels[i])):
           accuracy += 1
       accuracy = accuracy/len(predictions)
       accuracies += [accuracy]
    return  w2,b2,w1,b1,accuracies

def test_mlp(w1,b1,w2,b2,data_test,labels_test):
    # Forward pass
    one_hot_labels = one_hot(labels_test)
    
   
    a0 = data_test # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = softmax(z2) # output of the output layer (BINARY CROSS ENTROPY activation function)
    predictions = a2  # the predicted values are the outputs of the output layer
    accuracy = 0
    for i in range(0,len(predictions)):
        if list(predictions[i]).index(max(predictions[i])) == list(one_hot_labels[i]).index(max(one_hot_labels[i])):
           accuracy += 1

    return accuracy/len(labels_test)




def run_mlp_training(data_train,labels_train,data_test,labels_test,d_h,learning_rate,num_epoch):
    
    d_in = len(data_train[0]) # 3072
    N = len(data_train) # 9900
    d_out = 10
    w1 = 2*np.random.rand(d_in, d_h) - 1
    w2 = 2*np.random.rand(d_h, d_out) - 1 
    b1 = np.zeros((1, d_h))  # first layer biaises
    b2 = np.zeros((1, d_out))  # second layer biaises
    
    w2,b2,w1,b1,accuracies = train_mlp(w1,b1,w2,b2,data_train,labels_train,learning_rate,num_epoch)
    final_testing_accuracy = test_mlp(w1,b1,w2,b2,data_test,labels_test)
    
    return accuracies,final_testing_accuracy

 


if __name__ == "__main__":

    #output = learn_once_cross_entropy(0.1)
    data, labels = read_cifar("data/cifar-10-batches-py")
    data_train, data_test, labels_train, labels_test = split_dataset(data,labels,split=0.01)
    accuracies,final_testing_accuracy = run_mlp_training(data_train,labels_train,data_test,labels_test,64,0.1,100)
    
    iterations = [i for i in range(0,100)]
    
    plt.plot(iterations,accuracies) 
  
    # naming the x axis 
    plt.xlabel('k values') 
    # naming the y axis 
    plt.ylabel('accuracy') 
  
    # giving a title to my graph 
    plt.title('accuracy variation according to k values') 
    plt.savefig("results/output-2.jpg")

    #print("final_testing_accuracy :",final_testing_accuracy)
    # function to show the plot 
    plt.show()

    

