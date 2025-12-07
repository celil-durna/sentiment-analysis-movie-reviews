# University Team-Project: Neural network model for sentiment analysis on movie reviews

Developed as a university project in a 2-person team using **pair programming**.

This README provides an overview of the project and its subtasks, describes the repository file structure, lists the dependencies required to run the code, explains how to run it, and summarizes our main results.


---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Repository File Structure](#repository-file-structure)  
3. [Dependencies](#dependencies)  
4. [How to run the code](#how-to-run-the-code)  
5. [Results](#results)  


---

## Project Overview

In this project, we developed a neural network that automatically classifies movie reviews with a rating from 1 to 10. The model is trained on a real-world dataset of 50,000 reviews, each containing both the review text and a numeric rating. Our goal is to build a model that learns from this data and then meaningfully classifies new, previously unseen reviews.

The main steps of the project are structured into seven subtasks:

- **1) Data preparation and Bag-of-Words**  
  Load the dataset, remove stopwords, split it into train/validation/test sets (80% / 10% / 10%), and create Bag-of-Words vectors using the 250 most frequent words in the training data.

- **2) Neural network initialization and forward pass**  
  Define the layer sizes (250 input features, 100 hidden neurons, 10 output neurons for ratings 1–10), build the `nn_model`, and test the forward propagation on the training data.

- **3) One-hot encoding and label checks**  
  Verify the one-hot encoding of the rating labels and check that the label arrays have the correct shapes and values.

- **4) Cost function and gradient check**  
  Compute the cross-entropy cost on the training set and run a gradient check to validate the backpropagation implementation.

- **5) Training the network**  
  Train the neural network using mini-batch stochastic gradient descent (batch size 500, up to 30 epochs) with a learning-rate schedule and track validation cost and accuracy.

- **6) Visualising cost and accuracy**  
  Plot the validation cost and validation accuracy over epochs to analyse training progress.

- **7) Prediction on validation data**  
  Use the trained model to predict ratings for the validation set and save the predictions to `val_predictions.csv`.


---

## Repository File Structure

- **`main.py`** – The main program to run the whole neural network pipeline (steps 1–7)
- **`methods.py`** – Data preparation & Bag-of-Words helper functions
- **`nn_model.py`** – Neural network implementation (build, train, predict, visualize)
- **`movie_review_data.csv`** – Movie review dataset (tracked with Git LFS)
- **`stopwords-en.txt`** – English stopword list used in preprocessing
- **`Ausarbeitung.pdf`** – Written project report (German)
- **`.gitattributes`** – Git LFS configuration for large files


---

## Dependencies

To run this project, you need:

- **Python** 3.10 or higher  
- **Git LFS** (the dataset `movie_review_data.csv` is tracked with LFS)  
  ```bash
  git lfs install
  ```

Python packages:

- `pandas`
- `numpy`
- `matplotlib`

You can install the required Python packages with:

```bash
pip install pandas numpy matplotlib
```


---

## How to run the code

1. Clone the repository (recommended: skip auto LFS download).
   
   To avoid issues while cloning, clone the repo with LFS auto-download disabled:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/celil-durna/sentiment-analysis-movie-reviews.git
   ```
   After cloning, `movie_review_data.csv` will usually be a small Git LFS pointer file (this is expected).

2. Navigate to the `sentiment-analysis-movie-reviews/` directory.

3. Get the dataset (choose one option).
   
   Option A: Download via Git LFS (recommended if the download finishes quickly for you)
   ```bash
   git lfs pull
   ```
   This downloads the real dataset file (~122 MB).

   Option B: Manual download (if `git lfs pull` is too slow)
   - Open the repository on GitHub in your browser
   - Click `movie_review_data.csv` and download it manually
   - Overwrite the small LFS pointer file (movie_review_data.csv) with the downloaded real CSV file

4. Run the main script with the following command:

   ```bash
   python main.py
   ```


---

## Results

On the validation set, the neural network tends to predict very extreme ratings (mostly 1 or 10), while the middle classes (2–9) are chosen much less frequently. The softmax outputs are often not clearly one-hot: the highest probability is only slightly larger than the others, which indicates some uncertainty. For clearly positive reviews the model usually assigns a high rating, but for negative reviews it behaves less reliably and sometimes still predicts a very high score.
