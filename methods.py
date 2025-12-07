import numpy as np
# from gensim.models import Word2Vec


# =============================================================================
#     Find the most frequent words in positive and negative reviews
# =============================================================================

def most_frequent_words(data, column, n):
    """
    Find the most frequent words in positive and negative reviews.
    
    Parameters
    ----------
    data : DataFrame 
        The DataFrame containing the reviews and their sentiment.
    column : string
        The name of the column containing the reviews.
    n : integer
        The number of most common words to return.
    
    Returns
    -------
    most common_pos : list of tuples
        The most common words in positive reviews and their counts.
    most common_neg : list of tuples
        The most common words in negative reviews and their counts.
    
    """
    # Separate reviews by sentiment
    data_neg = " ".join(data.loc[data['sentiment'] == 0, column]).split()
    data_pos = " ".join(data.loc[data['sentiment'] == 1, column]).split()

    # Count the words and get the most common ones
    word_count_neg = {}
    word_count_pos = {}

    for word in data_neg:
        word_count_neg[word] = word_count_neg.get(word, 0) + 1

    for word in data_pos:
        word_count_pos[word] = word_count_pos.get(word, 0) + 1

    most_common_neg = sorted(word_count_neg.items(), key=lambda x: x[1], reverse=True)[:n]
    most_common_pos = sorted(word_count_pos.items(), key=lambda x: x[1], reverse=True)[:n]

    return most_common_pos, most_common_neg


# =============================================================================
#     Remove stopwords
# =============================================================================

def remove_stopwords(data, column, stopwords_path):
    """
    Remove stopwords from the reviews in the DataFrame.
    
    Parameters
    ----------
    data : DataFrame 
        The DataFrame containing the reviews.
    column : string
        The name of the column containing the reviews.
    stopwords_path : string
        The path to the stopwords file.
        
    Returns
    -------
    data : DataFrame 
        The DataFrame with a new column 'text_wo_stoppwords' containing the reviews 
        without stopwords.
        
    """
    
    # Load stopwords from the specified file
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())

    # Remove stopwords from the 'text_cleaned' column
    data['text_wo_stopwords'] = data[column].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stopwords])
        )
    
    return data


# =============================================================================
#     Test Train Data Split
# =============================================================================

def generate_data(data, relevant_columns, p_val, p_test):
    """ 
    Splits the data into training, validation, and test sets.
    
    Parameters
    ----------
    data : DataFrame 
        The DataFrame containing the reviews and their sentiment.
    relevant_columns : list of strings
        The names of the columns to keep in the resulting DataFrames.
    p_val : float
        The proportion of the data to use for the validation set.
    p_test : float
        The proportion of the data to use for the test set.
    
    Returns
    -------
    train_data : DataFrame 
        The training set.
    val_data : DataFrame
        The validation set.
    test_data : DataFrame
        The test set.
        
    """
    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate the sizes of each split
    test_end = int(np.floor(p_test * len(data)))
    val_end = int(np.floor((p_test + p_val) * len(data)))

    # Split the data into train, validation, and test sets
    test_data = data[:test_end]
    val_data = data[test_end:val_end]
    train_data = data[val_end:]
    
    # only keep the columns we need
    train_data = train_data[relevant_columns]
    val_data = val_data[relevant_columns]
    test_data = test_data[relevant_columns]

    return train_data, val_data, test_data


# =============================================================================
#     Bag-of-Words Vocabulary Selection
# =============================================================================

def bow_set(data, column, n):
    """
    Find the n most frequent words in a text column.

    Parameters
    ----------
    data : DataFrame 
        The DataFrame containing the reviews.
    column : string
        The name of the column containing the cleaned reviews.
    n : integer
        The number of most common words to return.

    Returns
    -------
    list
        A list with the n most frequent words in the reviews.

    """
    # Combine all reviews into one string and split into words
    words = " ".join(data[column]).split()

    # Count the words
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    # Sort by frequency of the words and take the first n words
    most_common = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:n]

    # Return just the words
    return [word for word, _ in most_common]


# =============================================================================
#     Bag-of-Words Vectorization
# =============================================================================

def bow_vectorization(data, column, relevant_words):
    """
    Transforms text into Bag-of-Words vectors using the given vocabulary.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the text.
    column : str
        The name of the column containing the text to be vectorized.
    relevant_words : list[str]
        The list of words (vocabulary) to consider for vectorization.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with an additional column 'bow_vectorized'
        containing the BoW vector representation of each text.

    """
    # Create a dictionary that maps words to their index in the vector
    word_index = {word: index for index, word in enumerate(relevant_words)}

    def text_to_vector(text):
        vector = [0] * len(relevant_words) # null-vector of length of vocabulary
        for word in text.split():
            if word in word_index:
                vector[word_index[word]] += 1
        return vector

    # Add a new column with the BoW vector representation
    data['bow_vectorized'] = data[column].apply(text_to_vector) # "apply-function" applies a function on every row in DF
    
    return data
