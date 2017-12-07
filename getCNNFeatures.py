import numpy as np 
import pandas as pd 
import pickle
import nltk
#from collection import defaultdict
import re 
from bs4 import BeautifulSoup
import sys 
import os 
from nltk.corpus import stopwords

#os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.utils.np_utils import to_categorical 
from keras.layers import Embedding 
from keras.layers import Dense, Input, Flatten 
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout 
from keras.models import Model 

max_seq_length = 1000
max_nb_words = 20000
embedding_dim = 100
validation_split = 0.2 

def clean_str(string):
    """
    Tokenization (string cleaning) for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()

train = pd.read_csv('labelledTrainData.csv', sep='\t')
print(train.shape)
test = pd.read_csv('labelledTestData.csv', sep = '\t')
print(test.shape)

texts = []
labels = []

# preprocessing
for idx in range(train.blogs.shape[0]):
	# remove html tags from individual blogs
    text = BeautifulSoup(train.blogs[idx], "html.parser")
    # removing "\", "'" and '"' tags
    texts.append(clean_str(text.get_text()))
    labels.append(train.label[idx])

test_texts = []
test_labels = []

for idx in range(test.blogs.shape[0]):
	# remove html tags from individual blogs
    text = BeautifulSoup(test.blogs[idx], "html.parser")
    # removing "\", "'" and '"' tags
    test_texts.append(clean_str(text.get_text()))
    test_labels.append(test.label[idx])

print(len(texts), len(labels))
print(len(test_texts), len(test_labels))

# using keras.tokenizer for tokenizing the preprocessed blogs
tokenizer = Tokenizer(num_words= max_nb_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

	# pad all blogs to make length uniform, ie 1000
blogs = pad_sequences(sequences, maxlen = max_seq_length)

	# change labels to an array of two categories
labels = to_categorical(np.asarray(labels))
print("shape of train blogs tensor: ", blogs.shape)
print("shape of train label tensor: ", labels.shape)

indices= np.arange(blogs.shape[0])
np.random.shuffle(indices) # shuffle the blogs randomly
blogs = blogs[indices]
labels= labels[indices]
	

nb_validation_samples = int(validation_split * blogs.shape[0])

x_train = blogs[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = blogs[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


# using keras.tokenizer for tokenizing the preprocessed blogs
tokenizer = Tokenizer(num_words= max_nb_words)
tokenizer.fit_on_texts(test_texts)
sequences = tokenizer.texts_to_sequences(test_texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

	# pad all blogs to make length uniform, ie 1000
test_blogs = pad_sequences(sequences, maxlen = max_seq_length)

	# change labels to an array of two categories
test_labels = to_categorical(np.asarray(test_labels))
print("shape of test blogs tensor: ", test_blogs.shape)
print("shape of test label tensor: ", test_labels.shape)

test_indices= np.arange(test_blogs.shape[0])
np.random.shuffle(test_indices) # shuffle the blogs randomly
test_blogs = test_blogs[test_indices]
test_labels= test_labels[test_indices]
	

nb_validation_samples = int(validation_split * blogs.shape[0])

x_test = test_blogs[:]
y_test = test_labels[:]

print("test blogs length: ",len(x_test))
print("test labels length ", len(y_test))

print("no of female and male blog authors in train and validation set: \n")
print(y_train.sum(axis = 0))
print(y_val.sum(axis = 0))

# for randomizing the vector of glove for unknown word:
embeddings_index = {}
f = open('glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:])
	embeddings_index[word] = coefs
f.close()

print("Total %s word vectors in Glove 6B 100d." % len(embeddings_index))
'''
embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))

for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		#words not found in embedding index will be all-zeros
		embedding_matrix[i] = embedding_vector 

embedding_layer = Embedding(len(word_index) + 1, embedding_dim, 
					weights = [embedding_matrix], input_length = max_seq_length,
					trainable = True)

# a simple convolutional approach
sequence_input = Input(shape = (max_seq_length,), dtype = 'int32')
embedding_sequences = embedding_layer(sequence_input)
conv1 = Conv1D(128, 5, activation = 'relu')(embedding_sequences)
pool1 = MaxPooling1D(5)(conv1)
conv2 = Conv1D(128, 5, activation = 'relu')(pool1)
pool2 = MaxPooling1D(5)(conv2)
conv3 = Conv1D(128, 5, activation = 'relu')(pool2)
pool3 = MaxPooling1D(35)(conv3)
flat = Flatten()(pool3)
dense = Dense(128, activation = 'relu')(flat)
preds = Dense(2, activation= 'softmax')(dense)

model = Model(sequence_input, preds)
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])
print(" ###############  Model fitting  ###################")
model.summary()
model.fit(x_train, y_train, validation_data = (x_val, y_val), nb_epoch = 15, batch_size = 128)

score = model.evaluate(x_test, y_test, batch_size = 32)
print("########Raw test score with simple cnn: ", score)
'''
embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))

for word , i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1, embedding_dim, 
					weights = [embedding_matrix], input_length = max_seq_length,
					trainable = True)

# a more complex convolutional approach
convs = [] 
filter_sizes = [3,4,5]

sequence_input = Input(shape = (max_seq_length,), dtype= 'int32')
embedding_sequences = embedding_layer(sequence_input)

for i in filter_sizes:
	conv_l1 = Conv1D(nb_filter = 128, filter_length = i, activation = 'relu')(embedding_sequences)
	l_pool = MaxPooling1D(5)(conv_l1)
	convs.append(l_pool)

l_merge = Merge(mode = 'concat', concat_axis = 1)(convs)
conv1 = Conv1D(128, 5, activation = 'relu')(l_merge)
pool1 = MaxPooling1D(5)(conv1)
conv2 = Conv1D(128, 5, activation = 'relu')(pool1)
pool2 = MaxPooling1D(30)(conv2)
flat = Flatten()(pool2)
dense = Dense(128, activation = 'relu')(flat)
preds = Dense(2, activation = 'softmax')(dense)

model = Model(sequence_input, preds)
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])

print(" $$$$$$$$$$$$$$ Deeper model fitting $$$$$$$$$$$$$$$")
#model.summary()
model.fit(x_train, y_train, validation_data = (x_val, y_val), nb_epoch = 200, batch_size = 50)

score = model.evaluate(x_test, y_test, batch_size = 50)
print("###########Raw test score with complex cnn : ", score)