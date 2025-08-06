#-----------------------------------------------------------------------------------
# In this project you will develop a neural network with fully-connected layers
# to perform classification, and test it out on the CIFAR-10 dataset.
#-----------------------------------------------------------------------------------

# setup
import numpy as np 
import matplotlib.pyplot as plt 
from simple_nn import SimpleNet 
from gradient_check import eval_numerical_gradient 
from data_utils import get_CIFAR10_data 
from vis_utils import visualize_grid


#-------------------------------------------------------
# Helper functions
# ------------------------------------------------------

def rel_error(x, y):
    """ Returns relative error """

    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def show_net_weights(net):
    """ Shows net weights """

    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


# Create a small net and some toy data to check your implementations.
# Note that the randoms seed are set for repeatable experiments.

input_size  = 4
hidden_size = 10
num_classes = 3
num_inputs  = 5

def init_toy_model():
    np.random.seed(0)
    return SimpleNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()

#-------------------------------------------------------
# Implement a Neural Network 
# ------------------------------------------------------
#
# Implement the first part of the forward pass in the function net.loss() of the
# class SimpleNet, which uses the weights and biases to compute the scores for
# all inputs.

scores = net.loss(X)
print('Your scores:')
print(scores)
print('\ncorrect scores:')
correct_scores = np.asarray([
 [0.36446210, 0.22911264, 0.40642526],
 [0.47590629, 0.17217039, 0.35192332],
 [0.43035767, 0.26164229, 0.30800004],
 [0.41583127, 0.29832280, 0.28584593],
 [0.36328815, 0.32279939, 0.31391246]])
print(correct_scores,'\n')

# The difference should be very small. (< 1e-7)
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))



# Forward pass: compute the loss. In the same function, implement the second
# part that computes the data and regularization loss.

loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133

# The difference should be very small. (< 1e-12)
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))

# Implement the rest of the function. This will compute the gradient of the loss
# with respect to the variables `W1`, `b1`, `W2`, and `b2`.

# Use numeric gradient checking to check your implementation of the backward
# pass.  
loss, grads = net.loss(X, y, reg=0.05)

# These should all be less than 1e-8 or so
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))



#-------------------------------------------------------
# Train the network
# ------------------------------------------------------
#
# Train the network with the stochastic gradient descent (SGD) strategy. Look at
# the function `SimpleNet.train` and fill in the missing sections to implement
# the training procedure.  You will also have to implement
# `SimpleNet.predict`.
#
# Once you have implemented the method, run the code below to train a two-layer
# network on toy data. You should achieve a training loss smaller than
# 0.02.

net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=True)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.figure(1)
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()

# ------------------------------------------------------
# Load real data 
# ------------------------------------------------------
#
# Load the data: now that you have implemented a two-layer network that passes
# gradient checks and works on toy data, you can load CIFAR-10 data so you can
# use it to train a classifier on a real dataset. 
#
# Invoke the get_CIFAR10_data function to get the data.

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Visualize some images to get a feel for the data
plt.figure(2)
plt.imshow(visualize_grid(X_train[:100, :].reshape(100, 32,32, 3), padding=3).astype('uint8'))
plt.gca().axis('off')
plt.show()

# Create a neural network with the right configuration for CIFAR-10
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = SimpleNet(input_size, hidden_size, num_classes)

# Train a network with SGD. In addition, the learning rate will be adjusted with
# an exponential learning rate schedule as optimization proceeding: after each
# epoch, the learning rate will be reduced by multiplying it by a decay rate.
stats = net.train(X_train, y_train, X_val, y_val, num_iters=1000,
                  batch_size=200, learning_rate=1e-4, learning_rate_decay=0.95,
                  reg=0.25, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)
exit(-1)
# ------------------------------------------------------
# Debug the training
# ------------------------------------------------------
#
# With the default parameters we provided above, you should get a validation
# accuracy of about 0.29 on the validation set, which is not very good.
#
# One strategy for getting insight into what is wrong is to plot the loss
# function and the accuracies on the training and validation sets during
# optimization.
#
# Another strategy is to visualize the weights that were learned in the first
# layer of the network. In most neural networks trained on visual data, the
# first layer weights typically show some visible structure when visualized.


# Plot the loss function and train / validation accuracies
plt.figure(3)
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.show()


# Visualize the weights of the network
plt.figure(5)
show_net_weights(net)

import itertools

# Grid of hyperparameters
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
regularization_strengths = [0.25, 0.5, 1.0]
hidden_sizes = [100, 200, 400]

best_val_acc = -1
results = {}

for hs, lr, reg in itertools.product(hidden_sizes, learning_rates, regularization_strengths):
    print(f'Training with hidden_size={hs}, learning_rate={lr}, reg={reg}')
    
    net = SimpleNet(input_size, hs, num_classes)
    stats = net.train(X_train, y_train, X_val, y_val,
                      num_iters=2000,
                      batch_size=200,
                      learning_rate=lr,
                      learning_rate_decay=0.95,
                      reg=reg,
                      verbose=False)
    
    val_acc = (net.predict(X_val) == y_val).mean()
    train_acc = (net.predict(X_train) == y_train).mean()
    results[(hs, lr, reg)] = (train_acc, val_acc)
    
    print(f'Train acc: {train_acc}, Val acc: {val_acc}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_net = net
# ------------------------------------------------------
# Tune your hyperparameters
# ------------------------------------------------------
#
# What's wrong? Looking at the visualizations above, you can see that the
# loss is decreasing more or less linearly, which seems to suggest that the
# learning rate may be too low. Moreover, there is no gap between the training
# and validation accuracy, suggesting that the model used has low capacity, and
# that its size should be increased. On the other hand, with a very large model
# you would expect to see more overfitting, which would manifest itself as
# a very large gap between the training and validation accuracy.
#
# Tuning the hyperparameters and developing intuition for how they
# affect the final performance is a large part of using Neural Networks, so you
# want to get a lot of practice. Below, you should experiment with different
# values of the various hyperparameters, including hidden layer size, learning
# rate, number of training epochs, and regularization strength. You might also
# consider tuning the learning rate decay, but you should be able to get good
# performance using the default value.
#
# Approximate results. You should aim to achieve a classification accuracy
# of greater than 48% on the validation set. 
#
# Experiment: Your goal in this exercise is to get as good of a result on
# CIFAR-10 as you can (52% could serve as a reference), with a fully-connected
# Neural Network. Feel free to implement your own techniques (e.g. PCA to reduce
# dimensionality, or adding dropout, or adding features to the solver, etc.).

# Explain your hyperparameter tuning process in a separate write-up.

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
#################################################################################

best_net = None # store the best model into this

# Tune the hyperparameters and add your own techniques
# ***** your code here *****
pass
# ***** end of your code *****

# visualize the weights of the best network
plt.figure(6)
show_net_weights(best_net) 

# Run on the test set
# When you are done experimenting, you should evaluate your final trained
# network on the test set; you should get above 48%.
test_acc = (best_net.predict(test_pca) == y_test).mean()
print('Test accuracy: ', test_acc)
