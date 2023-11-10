import numpy as np
from collections import Counter
import matplotlib.pyplot as plt 
from read_cifar import read_cifar, split_dataset


def distance_matrix(A,B):
    return np.sum((A[:,None] - B)**2, axis=-1)**.5


def knn_predict(dists,labels_train,K):
    labels = []
    for i in range(0,len(dists)):
         idx = np.argpartition(dists[i], K)
         neighbors_labels = [labels_train[j] for j in idx]
         c = Counter(neighbors_labels)
         neighbors_labels_count = c.items()
         Max = 0
         test_data_label = None
         for lab in neighbors_labels_count:
             if lab[1] > Max:
                Max = lab[1]
                test_data_label = lab[0]
         labels += [int(test_data_label)]
    return np.asarray(labels)

def evaluate_knn(data_train,labels_train,data_test,labels_test,k):
    dists = distance_matrix(data_test,data_train)
    print(dists.shape)
    testing_data_labels = knn_predict(dists,labels_train,k)
    print(labels_test)
    accuracy = 0
    for i in range(0,len(testing_data_labels)):
        if testing_data_labels[i] == labels_test[i]:
            accuracy += 1
    return accuracy/len(testing_data_labels)

def plot_accuracy_k(K_values,data_train,labels_train,data_test,labels_test):
    k_accuracy_value = []
    for k in K_values:
       k_accuracy_value += [evaluate_knn(data_train,labels_train,data_test,labels_test,k)]  
    
    # plotting the points  
    plt.plot(K_values,k_accuracy_value) 
  
    # naming the x axis 
    plt.xlabel('k values') 
    # naming the y axis 
    plt.ylabel('accuracy') 
    
    # giving a title to my graph 
    plt.title('accuracy variation according to k values') 
    plt.savefig("results/output-1.jpg")

    # function to show the plot 
    plt.show()

if __name__ == "__main__":

    ## Les ressources locale de calcul me permettent de traiter un seul batch à la fois.le choix de split 0.01 1% du batch sera considéré parmi les données de test (99% du batch sera utilisé pour l'entrainement), ce choix a été forcé par les calculs. En choississant un taux supérieur à 1%, les ressources de calcul local ne peuvent plus subvenir aux besoins de calcul notement en terme de mémoire.  
    data, labels = read_cifar("data/cifar-10-batches-py")
    data_train, data_test, labels_train, labels_test = split_dataset(data,labels,split=0.01)
    K_values = [i for i in range(1,21)]
    plot_accuracy_k(K_values,data_train,labels_train,data_test,labels_test)

    


        

         
    

        
    
   
