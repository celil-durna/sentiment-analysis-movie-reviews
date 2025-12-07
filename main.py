import pandas as pd
from methods import remove_stopwords, generate_data, bow_set, bow_vectorization
from nn_model import nn_model
import numpy as np


# =============================================================================
#     1) Prepare Bag-of-Words input data (load, clean, split, vectorize)
# =============================================================================
print("\nTask 1 - Prepare Bag-of-Words input data:\n")

# Load data
df = pd.read_csv("movie_review_data.csv")

# Remove stopwords
df = remove_stopwords(df, 'text_clean', 'stopwords-en.txt')

# Relevant columns
relevant_columns = ['sentiment', 'rating', 'text_wo_stopwords']

# Split into train, validation, and test datasets
train_data, val_data, test_data = generate_data(df, relevant_columns, p_val=0.1, p_test=0.1)

# Choose number of most frequent words
n = 250

# Get vocabulary based on training set
vocab = bow_set(train_data, 'text_wo_stopwords', n)

# Vectorize all datasets
train_data = bow_vectorization(train_data, 'text_wo_stopwords', vocab)
val_data = bow_vectorization(val_data, 'text_wo_stopwords', vocab)
test_data = bow_vectorization(test_data, 'text_wo_stopwords', vocab)

# Save to CSV
train_data.to_csv("train_bow.csv", index=False)
val_data.to_csv("val_bow.csv", index=False)
test_data.to_csv("test_bow.csv", index=False)

print("BoW-vectorized data saved\n\n\n")


# =============================================================================
#     2) Initialize neural network and test forward propagation
# =============================================================================
print("Task 2 - Initialize neural network and test forward propagation:\n")

# Define layer sizes:
# Input layer: 250 neurons, Hidden layer: 100 neurons, Output layer: 10 neurons (for ratings 1–10)
sizes = [250, 100, 10]

# Initialize the neural network
model = nn_model(train_data, val_data, test_data, sizes)

# Test forward propagation after initialization 
print("\nTesting forward propagation:")
X_sample = np.array(train_data['bow_vectorized'].tolist()) # Convert DF column to NumPy array
A, Z = model.nn_forward(X_sample) # Run forward propagation
# Print overview of shapes and sample values
for l in range(model.num_layers - 1):
    print('\nLayer %i -> Layer %i' % (l, l + 1))
    print('A[%i] shape                        : %s' % (l, str(A[l].shape)))
    print('A[%i] sample (first row, 5 values) : %s' % (l, str(A[l][0][:5])))
    print('W[%i] shape                        : %s' % (l, str(model.weights[l].shape)))
    print('b[%i] shape                        : %s' % (l, str(model.biases[l].shape)))
    print('Z[%i] shape                        : %s' % (l, str(Z[l].shape)))
    print('Z[%i] sample (first row, 5 values) : %s' % (l, str(Z[l][0][:5])))
    print('A[%i] shape (after activation)     : %s' % (l+1, str(A[l+1].shape)))
    print('A[%i] sample (first row, 5 values) : %s' % (l+1, str(A[l+1][0][:5])))
# Final output prediction (rating 1-10)
print('\nOutput prediction (softmax result for first sample):')
print(A[-1][0]) # -1 for last index -> activation in output layer
print('-> Predicted rating class :', A[-1][0].argmax() + 1) # argmax() gives index of highest value in array

print("\nNeural network initialized\n\n\n")


# =============================================================================
#     3) Test One-Hot-Encoding and label arrays 
# =============================================================================
print("Task 3 - Test One-Hot-Encoding and label arrays:\n")

# Print shapes of the encoded label arrays
print("Y_train shape:", model.Y_train.shape)
print("Y_val shape  :", model.Y_val.shape)
print("Y_test shape :", model.Y_test.shape)

# Run one-hot encoding manually (for training data) for testing correctness 
manual_Y_train = model.nn_one_hot(train_data['rating'].values, sizes)

# Select the first example (true rating for first text)
true_rating = train_data['rating'].values[0]
encoded_vector = manual_Y_train[0] # vector from one-hot encoding

# Print true rating and corresponding one-hot vector
print(f"\nFirst true rating                 :", true_rating)
print(f"One-hot vector from nn_one_hot()  :", encoded_vector)

# Create expected one-hot vector manually for comparison
expected_vector = np.zeros(sizes[-1])
expected_vector[true_rating - 1] = 1
print("Expected one-hot vector           :", expected_vector)

# Compare both vectors and print result
is_correct = np.array_equal(encoded_vector, expected_vector)
print("So is it correct?                 :", is_correct)

print("\nOne-hot encoding and label array test complete\n\n\n")


# =============================================================================
#     4) Test the cost function
# =============================================================================
print("Task 4 - Test the cost function:\n")

# Run forward propagation
A, Z = model.nn_forward(model.X_train)

# Compute and print cost
cost = model.nn_cost(model.Y_train, A[-1])
print("Training cost (first forward pass):", cost)

# Gradient Check
print("\nRunning gradient check via nn_check():")
model.nn_check()

print("\n\n")


# =============================================================================
#     5) Train the neural network
# =============================================================================
print("Task 5 - Train the neural network:\n")

# training parameters
maxepoch = 30
batch_size = 500
K = 1000 
eta_0 = 2 * (10**(-1))
eta_K = 10**(-2)

# Train neural network
model.nn_train(batch_size=batch_size, maxepoch=maxepoch, K=K, eta_0=eta_0, eta_K=eta_K)

print("Training complete\n\n\n")


# =============================================================================
#     6) Visualize the cost progress
# =============================================================================
print("Task 6 - Visualize the cost progress:\n")

model.visualize_cost_progress()

print("Visualization complete\n\n\n")


# =============================================================================
#     7) Predict rating on validation data
# =============================================================================
print("Task 7 - Predict rating on validation data:\n")

# Apply prediction function on validation dataset and save to CSV
df_pred = model.nn_predict(val_data)
df_pred.to_csv("val_predictions.csv", index=False)

print("Prediction complete\n\n\n")
