
# coding: utf-8

# # Predict tags on StackOverflow with linear models

# In[53]:


import sys
sys.path.append("..")


# In[54]:


from evaluate import Evaluate #evaluate instance is created for testing the trained topologies and for printing out the results


# In[55]:


evaluate = Evaluate()


# ### Text preprocessing

# In[56]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[57]:


'''
 In this task there is a dataset of post titles from StackOverflow. 
 Provided a split to 3 sets: train, validation and test. 
 All corpora (except for test) contain titles of the posts and corresponding tags (100 tags are available). 
 The test set is provided for testing the topologies. 
 Upload the corpora using pandas and look at the data:
'''
from ast import literal_eval
import pandas as pd
import numpy as np


# In[58]:


def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


# In[59]:


train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')


# In[60]:


train.head()


# In[61]:


X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values


# Task 1 (TextPrepare). 
# Implement the function text_prepare following the instructions. 
# After that, run the function test_test_prepare to test it on tiny cases.

# In[62]:


import re


# In[63]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwords from text
    return text


# In[64]:


def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


# In[65]:


print(test_text_prepare())


# In[66]:


prepared_questions = []
for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)
text_prepare_results = '\n'.join(prepared_questions)


evaluate.produce_tag('TextPrepare', text_prepare_results)


# Now we can preprocess the titles using function *text_prepare* and  making sure that the headers don't have bad symbols:

# In[67]:


X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]


# In[68]:


X_train[:3]


# For each tag and for each word calculate how many times they occur in the train corpus.
# Task 2 (WordsTagsCount). Find 3 most popular tags and 3 most popular words in the train data.

# In[69]:


from collections import defaultdict
# Dictionary of all tags from train corpus with their counts.
tags_counts = defaultdict(int) #{}
# Dictionary of all words from train corpus with their counts.
words_counts = defaultdict(int) #{}

for tags in y_train:
    for tag in tags:
        tags_counts[tag] +=1
for text in X_train:
    for word in text.split():
        words_counts[word] +=1


# In[70]:


most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

evaluate.produce_tag('WordsTagsCount', '%s\n%s' % (','.join(tag for tag, _ in most_common_tags), 
                                                ','.join(word for word, _ in most_common_words)))


# ### Transforming text to a vector
# 
# Machine Learning algorithms work with numeric data and we cannot use the provided text data "as is". There are many ways to transform text data to numeric vectors. In this task you will try to use two of them.
# 
# #### Bag of words
# 
# One of the well-known approaches is a *bag-of-words* representation. To create this transformation, follow the steps:
# 1. Find *N* most popular words in train corpus and numerate them. Now we have a dictionary of the most popular words.
# 2. For each title in the corpora create a zero vector with the dimension equals to *N*.
# 3. For each text in the corpora iterate over words which are in the dictionary and increase by 1 the corresponding coordinate.
# 
# Let's try to do it for a toy example. Imagine that we have *N* = 4 and the list of the most popular words is 
# 
#     ['hi', 'you', 'me', 'are']
# 
# Then we need to numerate them, for example, like this: 
# 
#     {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
# 
# And we have the text, which we want to transform to the vector:
# 
#     'hi how are you'
# 
# For this text we create a corresponding zero vector 
# 
#     [0, 0, 0, 0]
#     
# And iterate over all words, and if the word is in the dictionary, we increase the value of the corresponding position in the vector:
# 
#     'hi':  [1, 0, 0, 0]
#     'how': [1, 0, 0, 0] # word 'how' is not in our dictionary
#     'are': [1, 0, 0, 1]
#     'you': [1, 1, 0, 1]
# 
# The resulting vector will be 
# 
#     [1, 1, 0, 1]
#    
# Implement the described encoding in the function *my_bag_of_words* with the size of the dictionary equals to 5000. To find the most common words use train data. You can test your code using the function *test_my_bag_of_words*.

# In[71]:


DICT_SIZE = 5000

INDEX_TO_WORDS = sorted(words_counts.keys(), key=lambda x: words_counts[x], reverse=True)[:DICT_SIZE]
WORDS_TO_INDEX = {word:i for i, word in enumerate(INDEX_TO_WORDS)}
ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    
    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


# In[72]:


def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


# In[73]:


print(test_my_bag_of_words())


# Now apply the implemented function to all samples (this might take up to a minute):

# In[74]:


from scipy import sparse as sp_sparse


# In[75]:


X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)


# Task 3 (BagOfWords). 
# _______________________
# 
# For the 11th row in X_train_mybag find how many non-zero elements it has.

# In[76]:


row = X_train_mybag[10].toarray()[0]
non_zero_elements_count = (row > 0).sum()

evaluate.produce_tag('BagOfWords', str(non_zero_elements_count))


# TF-IDF
# The second approach extends the bag-of-words framework by taking into account total frequencies of words in the corpora. 
# It helps to penalize too frequent words and provide better features space.

# In[77]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[78]:


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       token_pattern='(\S+)')
    
    
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)
    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


# Once you have done text preprocessing, always have a look at the results. Be very careful at this step, because the performance of future models will drastically depend on it. 
# 
# In this case, check whether you have c++ or c# in your vocabulary, as they are obviously important tokens in our tags prediction task:

# In[79]:


X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}


# In[80]:


tfidf_vocab['c++']


# If you can't find it, we need to understand how did it happen that we lost them? It happened during the built-in tokenization of TfidfVectorizer. Luckily, we can influence on this process. Get back to the function above and use '(\S+)' regexp as a *token_pattern* in the constructor of the vectorizer.  

# Now, use this transormation for the data and check again.

# In[81]:


tfidf_reversed_vocab[1976]


# ### MultiLabel classifier
# 
# As we have noticed before, in this task each example can have multiple tags. To deal with such kind of prediction, we need to transform labels in a binary form and the prediction will be a mask of 0s and 1s. For this purpose it is convenient to use [MultiLabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) from *sklearn*.

# In[82]:


from sklearn.preprocessing import MultiLabelBinarizer


# In[83]:


mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)


# Implement the function *train_classifier* for training a classifier. In this task we suggest to use One-vs-Rest approach, which is implemented in [OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) class. In this approach *k* classifiers (= number of tags) are trained. As a basic classifier, use [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). It is one of the simplest methods, but often it performs good enough in text classification tasks. It might take some time, because a number of classifiers to train is large.

# In[84]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition.nmf import NMF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[85]:


def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    
#     clf = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(16,8,8), learning_rate='adaptive', max_iter=2000,verbose=True))
#     clf = OneVsRestClassifier(SVC(max_iter=1000, verbose=True))
#     clf = OneVsRestClassifier(LogisticRegression())
#     clf = OneVsRestClassifier(AdaBoostClassifier())
    clf = OneVsRestClassifier(RidgeClassifier(normalize=True))
    clf.fit(X_train, y_train)
    return clf   


# Train the classifiers for different data transformations: *bag-of-words* and *tf-idf*.

# In[86]:


classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)


# Now you can create predictions for the data. You will need two types of predictions: labels and scores.

# In[87]:


y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)


# Now take a look at how classifier, which uses TF-IDF, works for a few examples:

# In[88]:


y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))


# ### Evaluation
# 
# To evaluate the results we will use several classification metrics:
#  - [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
#  - [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
#  - [Area under ROC-curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
#  - [Area under precision-recall curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) 
#  
# Make sure you are familiar with all of them. How would you expect the things work for the multi-label scenario? Read about micro/macro/weighted averaging following the sklearn links provided above.

# In[89]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


# In[90]:



def print_evaluation_scores(y_val, predicted):
    
    print(accuracy_score(y_val, predicted))
    print(f1_score(y_val, predicted, average='weighted'))
    print(average_precision_score(y_val, predicted))


# In[91]:


print('Bag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('Tfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


# In[92]:


from metrics import roc_auc
get_ipython().run_line_magic('matplotlib', 'inline')


# In[93]:


n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_mybag, n_classes)


# In[94]:


n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_tfidf, n_classes)


# **Task 4 (MultilabelClassification).** Once evaluated set up, experiment a bit with training the classifiers. recommendation:
# - compare the quality of the bag-of-words and TF-IDF approaches and chose one of them.
# - for the chosen one, try *L1* and *L2*-regularization techniques in Logistic Regression with different coefficients (e.g. C equal to 0.1, 1, 10, 100).

# In[95]:


test_predictions = classifier_tfidf.predict(X_test_tfidf) ######### YOUR CODE HERE #############
test_pred_inversed = mlb.inverse_transform(test_predictions)

test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))
evaluate.produce_tag('MultilabelClassification', test_predictions_for_submission)


# ### Analysis of the most important features

# Finally, it is usually a good idea to look at the features (words or n-grams) that are used with the largest weigths in your logistic regression model.

# Implement the function *print_words_for_tag* to find them. Get back to sklearn documentation on [OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) and [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) if needed.

# In[96]:


def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary
        
        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))
    
    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator. 
    
    est = classifier.estimators_[tags_classes.index(tag)]
    top_positive_words = [index_to_words[index] for index in est.coef_.argsort().tolist()[0][-5:]]  # top-5 words sorted by the coefficiens.
    top_negative_words = [index_to_words[index] for index in est.coef_.argsort().tolist()[0][:5]] # bottom-5 words  sorted by the coefficients.
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))


# In[97]:


print_words_for_tag(classifier_tfidf, 'c', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'c++', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'linux', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)

