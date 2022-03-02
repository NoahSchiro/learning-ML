# MNIST Analysis

Intial parameters:

Hidden layer #: 1

Hidden layer size: 128

Activation function: relu

Optimizer: adam

Loss function: Sparse Categorical Cross entropy

Epochs: 10

Test accuracy: 97.53%

### Test 1: Increase nodes

Hidden layer size 128 -> 1568

Test accuracy: 98.15%

Other notes: Took much longer. Number of nodes and running 
time seems to be about proportional. For the next test I will try 
adding the same number of nodes but put them in different layers


### Test 2: Increase layers

Hidden layer size remained the same. Increased hidden layers from 1 -> 5.
Total nodes would then be 640

Test accuracy: 97.26%

Some indication that we got problems with overfitting (training 
accuracy was 99.18%)


### Test 3: Use hyperparameters

Hyperparameters told us to use about 592 nodes and two layers, 6 epochs and we got a very similar 
accuracy to our intial tests. This is likely just the peek preformance we can eek out of this 
dataset.
