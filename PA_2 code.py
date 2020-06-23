#!/usr/bin/env python
# coding: utf-8
# Vasan, Godse, Bouranis

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

def loadfunction(file):
    set = pd.read_csv(file)
    bias = pd.DataFrame(np.repeat(1, len(set)))
    set_no_lab = pd.DataFrame(set.drop(columns = [set.columns[0]]))
    labels = np.array(set.iloc[:,0])
    finalset = np.array(pd.concat([bias, set_no_lab], axis = 1))
    for i in range(0,len(labels)):
        if labels[i] == 3:
            labels[i] = +1
        else:
            labels[i] = -1
    return(finalset, labels)

X_train, Y_train = loadfunction('./pa2_train.csv')
X_val, Y_val = loadfunction('./pa2_valid.csv')
test_set = pd.read_csv('./pa2_test_no_label.csv')
bias = pd.DataFrame(np.repeat(1, len(test_set)))
X_test = np.array(pd.concat([bias, test_set], axis = 1))

class OnlinePerceptron(object):
    def __init__(self, iters = 15):
        self.iters = iters

    def predict(self, X, weights):
        predictions = np.sign(np.dot(X, weights.T))
        return predictions
            
    def train(self, X, labels, X_val, labels_val):

        train_accuracies = []
        val_accuracies = []
        weight_history = []

        weights = np.zeros(len(X.T))
        for itr in range(self.iters):
            for i in range(0,labels.shape[0]):
                xt = X[i]
                yt = labels[i]
                val = np.dot(yt,np.dot(xt, weights.T))
                if val <= 0:
                    wt = weights+np.dot(yt,xt)
                    weights = wt
                    
            # train and val predictions
            train_predictions = self.predict(X, weights)
            val_predictions = self.predict(X_val, weights)
            
            # train and val mistake count
            train_mistakes = len(train_predictions[train_predictions != labels])
            val_mistakes = len(val_predictions[val_predictions != labels_val])
            
            # train and val accuracies
            train_accuracies.append(len(train_predictions[train_predictions == labels]) / labels.shape[0] * 100)
            val_accuracies.append(len(val_predictions[val_predictions == labels_val]) / labels_val.shape[0] * 100)
        
            # weight history
            weight_history.append(weights)
            
            # logging information
            print("itr: {:3d}\t train (acc., mistakes): {:.2f}%\t{}\t val (acc., mistakes): {:.2f}%\t{}\t".format(itr, train_accuracies[-1], train_mistakes, val_accuracies[-1], val_mistakes))
            
        return np.array(weights), np.array(train_accuracies), np.array(val_accuracies), np.array(weight_history)




model = OnlinePerceptron(iters = 15)      
final_weights, train_accuracies, val_accuracies, weight_history = model.train(X_train, Y_train, X_val, Y_val)


model_final = OnlinePerceptron(iters = 6)
trained_weights, trained_accuracies, val_acc, weight_hist = model_final.train(X_train, Y_train, X_val, Y_val)
oplabel = model_final.predict(X_test, trained_weights)
pd.DataFrame(oplabel).to_csv(path_or_buf='oplabel.csv', index= False, header = False)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Training Accuracy vs Validation Accuracy')

ax.plot(np.arange(0,train_accuracies.shape[0]), train_accuracies, label='Train')
ax.plot(np.arange(0,val_accuracies.shape[0]), val_accuracies, label='Dev')
ax.set_xlabel('Iterations')
ax.set_ylabel('Accuracy')
ax.set_ylim([90,100])
ax.legend()
plt.show()
fig.savefig('onlinePerceptronTrainingVal.png')

# Get weights for model's best performance on validation set
idx = np.argmax(val_accuracies)
print("best weights at iteration {}".format(idx))
best_online_weights = weight_history[idx]



class AveragePerceptron(object):
    def __init__(self, iters = 15):
        self.iters = iters

    def predict(self, X, weights):
        predictions = np.sign(np.dot(X, weights.T))
#         print(predictions)
        return predictions
            
    def train(self, X, labels, X_val, labels_val):

        train_accuracies = []
        val_accuracies = []
        weight_history = []

        weights = np.zeros(len(X.T))
        avgweights = np.zeros(len(X.T))
        counter = 1
        for itr in range(self.iters):
           
            for i in range(0,labels.shape[0]):
                xt = X[i]
                yt = labels[i]
                val = np.dot(yt,np.dot(xt, weights.T))
                if val <= 0:
                    wt = weights+np.dot(yt,xt)
                    weights = wt
                avgweights = (counter*avgweights + weights)/(counter+1)
                counter = counter + 1
                
            # train and val predictions
            train_predictions = self.predict(X, avgweights)
            val_predictions = self.predict(X_val, avgweights)
            
            # train and val mistake count
            train_mistakes = len(train_predictions[train_predictions != labels])
            val_mistakes = len(val_predictions[val_predictions != labels_val])
            
            # train and val accuracies
            train_accuracies.append(len(train_predictions[train_predictions == labels]) / labels.shape[0] * 100)
            val_accuracies.append(len(val_predictions[val_predictions == labels_val]) / labels_val.shape[0] * 100)
        
            # weight history
            weight_history.append(avgweights)
            
            # logging information
            print("itr/counter: {:3d} / {:10d}\t train (acc., mistakes): {:.2f}%\t{}\t val (acc., mistakes): {:.2f}%\t{}\t".format(itr, counter, train_accuracies[-1], train_mistakes, val_accuracies[-1], val_mistakes))
            
        return np.array(weights), np.array(train_accuracies), np.array(val_accuracies), np.array(weight_history)




model = AveragePerceptron(iters = 15)      
final_weights, train_accuracies, val_accuracies, weight_history = model.train(X_train, Y_train, X_val, Y_val)

final_model = AveragePerceptron(iters = 8)      
trained_weights_ap, trained_acc, val_acc, weight_hist = final_model.train(X_train, Y_train, X_val, Y_val)
aplabel = final_model.predict(trained_weights_ap, X_test)
pd.DataFrame(aplabel).to_csv(path_or_buf='aplabel.csv', index= False, header = False)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Training Accuracy vs Validation Accuracy')

ax.plot(np.arange(0,train_accuracies.shape[0]), train_accuracies, label='Train')
ax.plot(np.arange(0,val_accuracies.shape[0]), val_accuracies, label='Dev')
ax.set_xlabel('Iterations')
ax.set_ylabel('Accuracy')
ax.set_ylim([90,100])
ax.legend()
plt.show()
fig.savefig('avgPerceptronTrainingVal.png')


# Get weights for model's best performance on validation set
idx = np.argmax(val_accuracies)
print("best weights at iteration {}".format(idx))
best_avg_weights = weight_history[idx]


X_train, Y_train = loadfunction('./pa2_train.csv')
X_val, Y_val = loadfunction('./pa2_valid.csv')
test_set = pd.read_csv('./pa2_test_no_label.csv')
bias = pd.DataFrame(np.repeat(1, len(test_set)))
X_test = np.array(pd.concat([bias, test_set], axis = 1))

def polynomial_kernel(x1, x2, p = 1):
    return np.power(np.dot(x1, x2), p)

class KernelPerceptron():
    def __init__(self, X, iters = 15, p = 1):
        self.iters = iters
        
        n_samples, n_features = X.shape
        
        # pre compute gram matrix
        n_samples = X_train.shape[0]
        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.K[i,j] = polynomial_kernel(X_train[i],X_train[j], p=p)
        print("Done computing gram matrix")
    
    def fit(self, xtrain, labels, xval, val_labels):
        n_samples, n_features = xtrain.shape
        
        self.alphas = np.zeros(n_samples)
        
        self.X = xtrain
        self.labels = labels
        
        train_accuracies = []
        val_accuracies = []
        alphas_history = []
                
        # training loop
        for itr in range(self.iters):
            
            for i in range(0, n_samples):

                u = np.sum(self.K[:,i] * self.alphas * self.labels)
    
                # if they are not the same sign
                if u * self.labels[i] <= 0:
                    self.alphas[i] += 1
                    
            # train and val predictions
            train_predictions = self.predict(xtrain)
            val_predictions = self.predict(xval)

            # train and val mistake count
            train_mistakes = len(train_predictions[train_predictions != self.labels])
            val_mistakes = len(val_predictions[val_predictions != val_labels])
                 
            # train and val accuracies
            train_accuracies.append(len(train_predictions[train_predictions == self.labels]) / self.labels.shape[0] * 100)
            val_accuracies.append(len(val_predictions[val_predictions == val_labels]) / val_labels.shape[0] * 100)
        
            # weight history
            alphas_history.append(self.alphas)
            
            # logging information
            print("itr: {:3d}\t train (acc., mistakes): {:.2f}%\t{}\t val (acc., mistakes): {:.2f}%\t{}\t".format(itr, train_accuracies[-1], train_mistakes, val_accuracies[-1], val_mistakes))
            
        return np.array(self.alphas), np.array(train_accuracies), np.array(val_accuracies), np.array(alphas_history)
    
    def predict(self, x):
        predictions = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            predictions[i] = np.sign( np.sum(self.K[:,i] * self.alphas * self.labels))
        return predictions




best_val_acc_for_ps = []

for p in [1,2,3,4,5]:
    model = KernelPerceptron(X_train, iters=15, p=p)
    alphas, train_accuracies, val_accuracies, alphas_history = model.fit(X_train, Y_train, X_val, Y_val)
    y_out = model.predict(X_val)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('[p = {}] Training Accuracy vs Validation Accuracy'.format(p))

    ax.plot(np.arange(1,train_accuracies.shape[0] + 1), train_accuracies, label='Train')
    ax.plot(np.arange(1,val_accuracies.shape[0] + 1), val_accuracies, label='Dev')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([55,100])
    ax.legend()
    plt.show()
    fig.savefig('kernelPerceptronTrainingVal{}.png'.format(p))

#  # Get weights for model's best performance on validation set
    idx = np.argmax(val_accuracies)
    print("best weights at iteration {}".format(idx))
    
    best_avg_weights = alphas_history[idx]
    best_val_acc_for_ps.append(val_accuracies[idx])

final_model_k = KernelPerceptron(X_train, iters=15, p=3)
trained_alphas, train_accs, val_accs, alphas_histry_trained = final_model_k.fit(X_train, Y_train, X_val, Y_val)
kplabel = final_model_k.predict(X_test)
pd.DataFrame(kplabel).to_csv(path_or_buf='kplabel.csv', index= False, header = False)
