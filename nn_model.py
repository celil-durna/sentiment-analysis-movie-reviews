import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt


class nn_model:
    
    def __init__(self, train_data, val_data, test_data, sizes):
        """
        Initialize the neural network model with the given parameters.
        Parameters
        ----------
        train_data : DataFrame
            DataFrame containing the training data.
        val_data : DataFrame
            DataFrame containing the validation data.
        test_data : DataFrame
            DataFrame containing the test data.
        sizes : list
            List of integers representing the sizes of each layer in the neural network.
        
        """
           
        self.sizes = sizes                              # Sizes of the layers of the neural network 
        self.num_layers = len(sizes)                    # Number of layers in the neural network
 
        # Neural network parameters - weights and biases
        self.biases = None             
        self.weights = None             

        # Test, validation and training data
        self.X_val = None
        self.X_test = None
        self.X_train = None
        self.Y_val = None
        self.Y_test = None
        self.Y_train = None         
        
        # do not assign data if the data is empty - required for nn.check function
        if not test_data.empty and not val_data.empty and not train_data.empty:
                    
            # Assign data to attributes
            # make sure that they are numpy arrays
            self.X_val = np.array(val_data['bow_vectorized'].tolist())
            self.X_test = np.array(test_data['bow_vectorized'].tolist())
            self.X_train = np.array(train_data['bow_vectorized'].tolist())

            # Assign labels to attributes 
            # make sure that they are one-hot encoded and numpy arrays
            self.Y_val = self.nn_one_hot(val_data['rating'].to_numpy(), sizes)
            self.Y_test = self.nn_one_hot(test_data['rating'].to_numpy(), sizes)
            self.Y_train = self.nn_one_hot(train_data['rating'].to_numpy(), sizes)
        
        self.nn_build(self.sizes, 0.12) 
             
 
# =============================================================================
#     Build neural network
# =============================================================================
    def nn_build(self, sizes, eps):
        """
        Builds neural network with layer sizes given in sizes. The first layer is
        the input layer and the last layer the output layer. The parameter eps 
        the standard deviation for the random (Gaussian mean 0) weight initialization. 
        The biases are initialized to zero.

        Parameters
        ----------
        sizes : list
            Layer sizes.
        eps : float
            Standard deviation for weight initialization.
        """
        
        # Randomly initialize weights from Gaussian distribution with 0 mean and eps standard deviation
        self.weights = [np.random.randn(sizes[0], sizes[1]) * eps]
        self.biases = [np.zeros(sizes[1])]
        
        for i in range(1, self.num_layers-1):
                self.weights.append(np.random.randn(sizes[i], sizes[i+1]) * eps)
                self.biases.append(np.zeros(sizes[i+1]))
        
        print('Neural network initialized with %i layers' % self.num_layers)
        print('Layer sizes: %s' % str(self.sizes))
        
        return 
    
     
# =============================================================================
#     One-hot-vector generation
# =============================================================================
    def nn_one_hot(self,Y,sizes):
        """
        Converts Y to one-hot encoding.
        The function also checks if the input is a 1D array and if not, raises an error.
        
        Parameters
        ----------
        Y : array
            Input data to be converted to one-hot encoding.
        sizes : list
            Sizes of the layers in the neural network. The last element is used to determine the number of classes.
        
        Returns
        -------
        Y_one_hot : array
            One-hot encoded array.
        """
        
        # Check if Y is a 1D array -> if not, raises an error
        if len(Y.shape) != 1:
           raise ValueError("Input Y must be a 1D array")
        
        # Number of samples (rows)
        num_samples = Y.shape[0] 

        # Number of output classes (from the last layer size)
        num_classes = sizes[-1]

        # Initialize One-hot matrix with zeros (shape: num_samples x num_classes)
        Y_one_hot = np.zeros((num_samples, num_classes))

        # Fill in 1s at correct class index 
        for i in range(num_samples):
            current_class = Y[i]
            Y_one_hot[i, current_class - 1] = 1 # adjusted by -1, since classes are 1-10
        
        return Y_one_hot
        

# =============================================================================
#     Forward pass through network 
# =============================================================================
    def nn_forward(self,X):
        """
        Forward propagation through neural network given by nn_model for data X.

        Parameters
        ----------
        X : array
            Data X.
            
        Returns
        -------
        A : list of arrays
            Activations.
        Z : list of arrays
            Weighted inputs.

        """

        # Get weights and biases and initialize Z and A

        # X = (1000, 250) for 1000 texts/reviews and 250 word-vector

        # W[0] = (250, 100)   Input -> Hidden
        # W[1] = (100, 10)    Hidden -> Output
        W = self.weights

        # b[0] = (100,1)   Hidden Layer
        # b[1] = (10,1)    Output Layer
        # 1 bias for every neuron
        b = self.biases

        # A[0] = [1000, 250]

        # Z[0] = [1000, 100]
        # A[1] = [1000, 100]

        # Z[1] = [1000, 10]
        # A[2] = [1000, 10]

        Z = [] 
        A = [X] # we start with the inputs
        
        # Loop through all layers
        for l in range(self.num_layers - 1):
            Z.append(A[l].dot(W[l]) + b[l])

            if l == self.num_layers - 2:
                A.append(self.softmax(Z[l]))  # last layer: apply Softmax
            else:
                A.append(self.sigmoid(Z[l]))  # for others: apply Sigmoid

        return A, Z
    

# =============================================================================
#     Neural network cost
# =============================================================================
    def nn_cost(self, Y, A_out):
        """
        Calculate cost function for neural network. Uses output from nn_forward.

        Parameters
        ----------
        Y : array
            True output.
        A_out : array
            Last layer of A from nn_forwards output.

        Returns
        -------
        cost : float
            Evaluation of cost function.

        """

        epsilon = 1e-8  # for numerical stability in log (because log(0) does not work)
        num_samples = Y.shape[0]  # number of samples
        log_probs = np.log(A_out + epsilon) # apply logarithmus on all softmax outputs (+ epsilon)
        cost = -np.sum(Y * log_probs) / num_samples # average cross-entropy loss formular: -1/M * sum(Y * log(A_out))

        return cost
    

# =============================================================================
#     Backward pass through network
# =============================================================================
    def nn_backward(self, X, Y, A, Z):
        """
        Backward propagation through neural network given by nn_model for data X, Y.
        Output from nn_forward is used.

        Parameters
        ----------
        X : array
            Data X.
        Y : array
            Data Y.
        A : list of arrays
            Activations.
        Z : list of arrays
            Weighted inputs.

        Returns
        -------
        W_gradients : list of arrays
            Gradients of weights.
        b_gradients : list of arrays
            Gradients of biases.

        """
        
        W = self.weights
        
        # Backpropagation 
        # Backpropagation for the last layer (output layer)
        # Derivative of categorical cross entropy cost function and softmax combined
        delta = [A[-1] - Y.reshape(A[-1].shape)]
        
        # Backpropagation for the prior layers
        for l in range(self.num_layers - 2, 0, -1):
            delta.append(delta[-1].dot(W[l].T) * self.sigmoid_gradient(Z[l-1])) 
           
        # Reverse for correct order   
        delta = delta[::-1]  
        
        # Calculate gradients
        W_gradients = []
        b_gradients = []
        
        for l in range(self.num_layers - 1):
            W_gradients.append(A[l].T.dot(delta[l]) / np.size(X, 0))
            b_gradients.append(np.sum(delta[l], axis=0) / np.size(X, 0))  
        
        # check for exploding or vanishing gradients
        if np.max(np.abs(W_gradients[-1])) > 1e5:
            print('Exploding gradient detected.')
            print(np.max(np.abs(W_gradients[-1])))
            
        if np.max(np.abs(W_gradients[-1])) < 1e-5:
            print('Vanishing gradient detected.')
            print(np.max(np.abs(W_gradients[-1])))
        
        return W_gradients, b_gradients     
        
    
# =============================================================================
#     Neural Network training
# =============================================================================
    def nn_train(self, batch_size, maxepoch, K, eta_0 = 0.01, eta_K = 1e-3, alpha = 0.001, beta1 = 0.9, beta2 = 0.999):
        """
        Stochastic gradient descent with replacement and update rule of learning rate
        from equation 8.14 in "Deep Learning".
        We use the old success rate and cost at current mini-batch for visualization of progress.
        This can be cheaply evaluated.

        Parameters
        ----------
        batch_size : integer
            Used mini-batch size. (number of text examples)
        maxepoch : integer
            Number of epochs for training. (how many times we go through all mini-batches)
        K : integer
            Learning rate parameter. (number of steps from eta_0 to eta_k)
        eta_0 : float
            Learning rate parameter. (starting learning rate)
        eta_K : float
            Learning rate parameter. (learning rate after k steps -> will be smaller)

        """

        # Task 6: lists for progress visualization
        self.costs = []  # to save validation costs after each epoch
        self.progress = []  # to save training costs after each epoch

        # Initial validation cost and success rate
        A_val, _ = self.nn_forward(self.X_val)
        val_cost = self.nn_cost(self.Y_val, A_val[-1])
        val_rate = self.nn_successrate(self.Y_val, A_val[-1])

        self.costs.append(val_cost)
        self.progress.append(val_rate)

        print(f"Initial validation cost: {val_cost:.4f}, accuracy: {val_rate:.4f}")
        
        # Number of samples (rows) in training data
        num_samples = self.X_train.shape[0] 

        # global step counter 
        # counts number of mini batches that we process (over all epoches)
        # used to reduce the learning rate step by step from eta_0 zu eta_K in k steps 
        k = 0  
    
        for epoch in range(maxepoch): # for all epoches do the following:

            # Shuffle all indices
            indices = list(range(num_samples))
            rnd.shuffle(indices)  
    
            # Create shuffled training dataset (-> random order)
            X_shuffled = self.X_train[indices]
            Y_shuffled = self.Y_train[indices]
    
            # Split into mini-batches
            # but we would have a rest: maybe we ignore last few samples due to integer division 
            #                           (but not bad because we shuffled before and just few samples)
            # we could also devide the samples by the formula "num_batches = (num_samples + batch_size - 1)" into batches
            # but with this method the last batch would be smaller
            num_batches = num_samples // batch_size 
    
            for i in range(num_batches): # for all mini-batches do the following:

                # Get mini-batch (by calculating start and end index in shuffled dataset for each mini-batch)
                start = i * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
    
                # Forward Propagation to get A and Z
                A, Z = self.nn_forward(X_batch)
                
                # Back Propagation to get gradients
                W_gradients, b_gradients = self.nn_backward(X_batch, Y_batch, A, Z)
    
                # Learning rate schedule
                # Update learning rate: linearly reduce from eta_0 (η_0) to eta_K (η_k) over K steps
                # The learning rate gets smaller step by step until k = K
                # After that it stays constant at eta_K
                if k <= K:
                    eta = (1 - k / K) * eta_0 + (k / K) * eta_K
                else:
                    eta = eta_K
    
                # Parameter update using gradient descent (Stochastisches Gradientenverfahren)
                # For each layer: adjust weights and biases in direction that reduces the cost
                # The learning rate (eta) controls how big each adjustment step is
                # W_gradients and b_gradients tell us how to change each parameter
                for l in range(self.num_layers - 1):
                    self.weights[l] -= eta * W_gradients[l]
                    self.biases[l]  -= eta * b_gradients[l]
    
                # Increment global step counter
                k += 1

            # Calculate costs and success rates
            A_val, _ = self.nn_forward(self.X_val)
            val_cost = self.nn_cost(self.Y_val, A_val[-1])
            val_rate = self.nn_successrate(self.Y_val, A_val[-1])

            # Save costs and rates for visualization
            self.costs.append(val_cost)
            self.progress.append(val_rate)

            print(
                f"Epoch {epoch + 1}/{maxepoch} - Val cost: {val_cost:.4f} | Val acc: {val_rate:.4f}")


# =============================================================================
#     Neural Network prediction
# =============================================================================
    def nn_successrate(self,Y, A_out):
        """
        Calculates prediction and success rate of neural network given data Y and
        results from nn_forward.

        Parameters
        ----------
        Y : array
            True output.
        A_out : array
            Last layer of A from nn_forwards output.

        Returns
        -------
        rate : float
            Success rate.

        """

        predictions = np.argmax(A_out, axis=1)
        targets = np.argmax(Y, axis=1)
        correct = np.sum(predictions == targets)
        rate = correct / len(Y)
        
        return rate 


# =============================================================================
#     progress visualization
# =============================================================================
    def visualize_cost_progress(self):
        """
        Parameters
            ----------
            None

            Returns
            -------
            None.
        """

        # x-Axis: Epochs (0 = initial untrained)
        epochs = range(len(self.costs))  # includes epoch 0

        # Plot validation cost
        plt.figure()
        plt.plot(epochs, self.costs, label="Validation Cost", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.title("Validation Cost over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot validation accuracy (progress)
        plt.figure()
        plt.plot(epochs, self.progress, label="Validation Accuracy", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy over Epochs (progress)")
        plt.legend()
        plt.grid(True)
        plt.show()

        return 
 

# =============================================================================
#     Network Predictions
# =============================================================================
    def nn_predict(self, df):
        """
        Predicts the output of the neural network given data X. The function uses
        the nn_forward function to calculate the output of the network.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing the data to be predicted. The dataframe must contain
            the vectorized data in the column specified by self.vectorization.

        Returns
        -------
        df : DataFrame
            DataFrame containing the original data and the predictions.

        """
        
        # Extract input matrix from "bow_vectorized" column 
        X = np.array(df['bow_vectorized'].tolist())  # shape: (num_samples, num_features)

        # Forward propagation with weights from training with training data
        A, _ = self.nn_forward(X)
        A_out = A[-1]  # outputs of neural networks (just last layer activations)
    
        # Compute predictions (class with highest probability -> rating 1-10)
        predictions = np.argmax(A_out, axis=1) + 1  # +1 because index start with 0
    
        # Add the 2 new columns to DataFrame
        df["A[-1]"] = A_out.tolist()        
        df["prediction"] = predictions      
    
        return df


#==============================================================================
#       Implementation check for neural network functions
#==============================================================================
    def nn_check(self):
        """
        Check to see if functions work correctly via gradient check. The functions
        nn_build, nn_forward, nn_cost, nn_backward are checked. A small random 
        neural network is generated and the gradient obtained by backpropagation 
        is compared to a numerical gradient.

        Parameters
        ----------
        None

        Returns
        -------
        None.

        """
        print('\n#       Gradient Checking       #\n')
        # Randomly initialize network from predefined ranges
        num_layers = 3 + rnd.randint(0, 4)
        sizes = [2 + rnd.randint(0, 30)] 
        for l in range(num_layers-2):
            sizes.append(2 + rnd.randint(0, 40)) 
        sizes.append(rnd.randint(2, sizes[0]))
        #sizes.append(10)
        nn_model_test = nn_model(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), sizes)
        
        # Generate X and Y randomly from predefined ranges
        m = 5 + rnd.randint(0, 50)
        X = np.random.randn(m, sizes[0])
        Y = (np.arange(m) % sizes[-1])
        Y = nn_model_test.nn_one_hot(Y, sizes)

        # Compute analytical gradients
        A, Z = nn_model_test.nn_forward(X)
        W_gradients, b_gradients = nn_model_test.nn_backward(X, Y, A, Z)
        
        # Initialize numerical gradients
        W_gradients_num = []
        b_gradients_num = []
        
        # Get weights
        W = nn_model_test.weights
        b = nn_model_test.biases
        
        # Compute numerical gradients of W by adding all possible perturbations
        perturbation = 1
        weights = []
        bias = []
        pert = []
        for i in np.arange(1, 6):
            for l in range(len(W)):
                W_gradients_num.append(np.zeros(np.shape(W_gradients[l])))
                for i in range(np.size(W[l], 0)):
                    for j in range(np.size(W[l], 1)):
                        nn_model_test.weights[l] = W[l]
                        nn_model_test.weights[l][i, j] = W[l][i, j] - perturbation 
                        A, Z = nn_model_test.nn_forward(X)
                        cost_minus = nn_model_test.nn_cost(Y, A[-1])
                        nn_model_test.weights[l] = W[l]
                        nn_model_test.weights[l][i, j] = W[l][i, j] + 2*perturbation 
                        A, Z = nn_model_test.nn_forward(X)
                        cost_plus = nn_model_test.nn_cost(Y, A[-1])
                        W_gradients_num[l][i, j] = (cost_plus - cost_minus) / (2*perturbation)
                        nn_model_test.weights[l][i, j] = W[l][i, j] - perturbation 
            
            # Compute numerical gradients of b by adding all possible perturbations            
            for l in range(len(b)):
                b_gradients_num.append(np.zeros(np.shape(b_gradients[l])))
                for i in range(np.size(b[l], 0)):
                    nn_model_test.biases[l] = b[l]
                    nn_model_test.biases[l][i] = b[l][i] - perturbation 
                    A, Z = nn_model_test.nn_forward(X)
                    cost_minus = nn_model_test.nn_cost(Y, A[-1])
                    nn_model_test.biases[l] = b[l]
                    nn_model_test.biases[l][i] = b[l][i] + 2* perturbation 
                    A, Z = nn_model_test.nn_forward(X)
                    cost_plus = nn_model_test.nn_cost(Y, A[-1])
                    b_gradients_num[l][i] = (cost_plus - cost_minus) / ( 2*perturbation)
                    nn_model_test.biases[l][i] = b[l][i] - perturbation 
            
            # Caculate norm differences
            b_diff = 0
            W_diff = 0
            for l in range(len(b)):
                W_diff += (np.linalg.norm(W_gradients_num[l] - W_gradients[l]) / 
                    np.linalg.norm(W_gradients_num[l] + W_gradients[l]))
                b_diff += (np.linalg.norm(b_gradients_num[l] - b_gradients[l]) / 
                    np.linalg.norm(b_gradients_num[l] + b_gradients[l]))
            
            # Save results
            weights.append(W_diff)
            bias.append(b_diff)
            pert.append(perturbation)
            perturbation *= 0.1

        # And print them to "see" the behaviour
        print('Pertubation and the correspondong errors in relative 2-norm')
        for i in np.arange(0, 5):
            print('pert={:5.3e} | weights={:5.3e} | biases={:5.3e}'.format(
                pert[i], weights[i], bias[i]))

        return


# =============================================================================
#     Sigmoid function
# =============================================================================
    def sigmoid(self, z):
        """
        Calculates componentwise sigmoid function at z.

        Parameters
        ----------
        z : float / array
            Input for sigmoid. Either float or array.

        Returns
        -------
        float / array
            Evaluation of sigmoid function at z.

        """
        
        # For very negative z (z-> -infty) a warning "Overflow" could occur.
        # can be avoided e.g. by clipping z to a range: 
        z = np.clip(z, -500, 500)
        
        return 1 / (1 + np.exp(-z) )


# =============================================================================
#     Softmax function
# =============================================================================
    def softmax(self, z):
        """
        Calculates softmax function at z.

        Parameters
        ----------
        z : float / array
            Input for softmax. Either float or array.

        Returns
        -------
        float / array
            Evaluation of softmax function at z.

        """
        
        # for numerical stability we subtract the max value of z from each element of z
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
        
        return e_z / np.sum(e_z, axis=1, keepdims=True)


# =============================================================================
#     Gradient of sigmoid function
# =============================================================================
    def sigmoid_gradient(self, z):
        """
        Calculates gradient of componentwise sigmoid function at z. 
        Uses sigmoid function.

        Parameters
        ----------
        z : float / array
            Input for sigmoid. Either float or array.

        Returns
        -------
        float / array
            Evaluation of gradient of sigmoid function at z.

        """
        
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    