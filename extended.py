import numpy as np
import pandas as pd
import re

import FMeasure
import wordClassFeatures
import genderPreferentialFeatures
import minePOSPats
import baseFeatures
import genderDifferencesFeatures
import get_CBOW_features
import elm 

from collections import defaultdict
from bs4 import BeautifulSoup

import sys
import os
import gc
#os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, Recurrent, TimeDistributed, Activation, BatchNormalization
from keras.layers import Dense, Input, Flatten, recurrent, wrappers, InputLayer
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM
from keras.layers import concatenate, GRU, Bidirectional
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint,  LearningRateScheduler
from keras import backend as K
from keras.utils import plot_model
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.regularizers import l2

from sklearn.svm import NuSVC, SVC
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from ggplot import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mode = 'TRAIN'
#mode = 'TEST'

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1

np.random.seed(0)

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = string.decode('utf-8')
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()

def findFeature(text):
	fMeasureFeature = FMeasure.FMeasure(text)
	genderPreferentialFeature = genderPreferentialFeatures.genderPreferentialFeatures(text)
	posFeature = minePOSPats.POSFeatures(text)
	baseFeature = baseFeatures.baseFeatures(text)
	genderDifferencesFeature = genderDifferencesFeatures.genderDifferencesFeatures(text)
	wordClassFeature = wordClassFeatures.wordClassFeatures(text)

	fMeasureToTuple = []
	featureVector = []
	fMeasureToTuple.append(fMeasureFeature)

	features =  genderDifferencesFeature + tuple(fMeasureToTuple) +\
			genderPreferentialFeature + posFeature + wordClassFeature +\
			baseFeature

	features = list(features)
	
	return features

data1 = pd.read_csv('original_blogs.csv', sep='\t')
print(data1.shape)

#a = input("pause")

texts = []
labels = []

#print(texts)

for idx in range(data1.Blog.shape[0]):
    text = BeautifulSoup(data1.Blog[idx], "html.parser")
    texts.append(clean_str(text.get_text().encode('ascii','ignore')))
    labels.append(data1.Gender[idx])

print(len(texts))
print(len(labels))

features = [findFeature(i) for i in texts]
#print(len(features[1]))
features = np.asarray(features, dtype=np.float32)
#print(features.shape)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
features = features[indices]


######################## PCA visualization #################
plt.cla()

print(features[1])
features = scale(features)
print(features[1])

pca = PCA(n_components=121)
features = pca.fit_transform(features)

variance = pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(var1)

'''
plt.plot(var1, color='red', linestyle='solid', linewidth=2.0)

plt.xlabel('No. of Principal Components')
plt.ylabel('Explained variance in percent')
plt.savefig('pca_features.png', dpi=150)
plt.show()
'''

###############  End of PCA visualization  ##########################
#b = input("pause")


nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-2*nb_validation_samples]
y_train = labels[:-2*nb_validation_samples]
train_features = features[:-2*nb_validation_samples]
#print(train_features.shape)

x_val = data[-2*nb_validation_samples:-nb_validation_samples]
y_val = labels[-2*nb_validation_samples:-nb_validation_samples]
val_features = features[-2*nb_validation_samples:-nb_validation_samples]

x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]
test_features = features[-nb_validation_samples:]

print('Traing and validation set number of positive and negative reviews:')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

GLOVE_DIR = "/home/saurav/Documents/gender verification/jan31/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))
'''
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
      
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

'''
class AttentionWithContext(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        # (x, 40, 300) x (300, 1)
        multData =  K.dot(x, self.kernel) # (x, 40, 1)
        multData = K.squeeze(multData, -1) # (x, 40)
        multData = multData + self.b # (x, 40) + (40,)

        multData = K.tanh(multData) # (x, 40)

        multData = multData * self.u # (x, 40) * (40, 1) => (x, 1)
        multData = K.exp(multData) # (X, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            multData = mask*multData #(x, 40) * (x, 40, )

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multData = K.expand_dims(multData)
        weighted_input = x * multData
        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
'''
######################## Simple model ###########################
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_lstm = Bidirectional(GRU(100, name='Bidirectional', return_sequences=True))(embedded_sequences)

l_lstm = AttentionWithContext()(l_lstm)
#ist_layer = K.expand_dims(time_dist_layer, axis=-1)

time_dist_layer = Dense(2, name='Dense')(l_lstm)

temp = K.expand_dims(time_dist_layer, axis=-1)
flat1 = Flatten()(temp)
#batch_norm_layer = BatchNormalization()(time_dist_layer)
outputs = Activation('softmax', name='Softmax_Activation')(time_dist_layer)

model = Model(sequence_input, outputs)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()

clf = SVC()
if mode == 'TRAIN':
	print("model fitting - attention GRU network")
	#plot_model(model, to_file='Simplest_LSTM_SVM_model.png')
	early_stop = EarlyStopping(patience=7)
	model.fit(x_train, y_train, validation_data=(x_val, y_val),
        epochs=1, batch_size=20,
		callbacks=[early_stop, ModelCheckpoint('best_checkpoint_simplest_GRU_attention_model.hdf5',
		save_best_only=True,verbose=1)])
	clf.fit(flat1, y_train)
	gc.collect()
	#model.save_weights('best_checkpoint.hdf5')


else:
	saved_weights = 'best_checkpoint_simplest_GRU_attention_model.hdf5'
	model.load_weights(saved_weights)
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Testing Accuracy: %.2f%%" % (scores[1]*100))

	scores = cross_val_score(clf, x_test , y_test, cv = 10)
	print("SVM accuracy:", sum(scores)/len(scores))
	print("############################")

####################### End of simple model #######################
'''
'''
################### Grid Search CV ###########################
train_features = train_features[:400]
print(train_features.shape)

def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(64, input_dim=(400,len(train_features[1])), kernel_initializer=init, activation='relu'))
	model.add(Dense(64, kernel_initializer=init, activation='relu'))
	#model.add(Dense(64, kernel_initializer=init, activation='sigmoid'))
	model.add(Dense(2, kernel_initializer=init, activation='softmax', W_regularizer=l2(0.01)))
	# Compile model
	model.compile(loss='hinge', optimizer=optimizer, metrics=['accuracy'])
	return model

########## Grid Search CV ###########

model = KerasClassifier(build_fn=create_model, verbose=0)
optimizers = ['rmsprop', 'adam', 'adadelta']
epochs=[20,30,40]
batches=[5,10,20]

init = ['glorot_uniform', 'normal', 'uniform']
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)

x_train = x_train[:400]
y_train = y_train[:400]

print("Grid searching!!")
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(train_features, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

################## End of Grid Search #########################
'''

def create_model():
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='sequence_input')
	embedded_sequences = embedding_layer(sequence_input)
	l_gru = Bidirectional(GRU(256, return_sequences=True))(embedded_sequences)

	l_gru = AttentionWithContext()(l_gru)

	auxiliary_output = Dense(2, activation='softmax', name='aux_output')(l_gru)

	auxiliary_input = Input(shape=(len(features[1]),), name='aux_input')
	combined = concatenate([l_gru, auxiliary_input])

	d1 = Dense(64, activation='relu')(combined)
	d2 = Dense(64, activation='relu')(d1)
	d3 = Dense(64, activation='relu')(d2)	
	
	#flat1 = Flatten()(d3)

	preds = Dense(2,  W_regularizer=l2(0.01))(d3)
	#flat2 = Flatten()(preds)
	#batch_norm_layer = BatchNormalization()(preds)
	main_output = Activation('softmax', name='main_output')(preds)

	model = Model(inputs=[sequence_input, auxiliary_input], outputs=[main_output,auxiliary_output])
	'''
	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'],
              loss_weights=[1., 0.2])
	'''
	model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
	
	return model

model= create_model()

if mode == 'TRAIN':
	print("model fitting - attention GRU network")
	model.summary()
	
	early_stop = EarlyStopping(patience=10)
	model.fit([x_train, train_features], [y_train,y_train], 
		validation_data=([x_val, val_features], [y_val,y_val]),
		epochs=50, batch_size=20, 
		callbacks=[early_stop, 
		ModelCheckpoint('best_checkpoint_features_GRU_SVM_pca.hdf5',
		save_best_only=True,verbose=1)])
	#model.save_weights('best_checkpoint.hdf5')


else:
	saved_weights = 'best_checkpoint_features_GRU_SVM_pca.hdf5'
	model.load_weights(saved_weights)
	plot_model(model, to_file='multi_input_model.png')
	scores = model.evaluate([x_test, test_features], [y_test, y_test], verbose=0)
	print("Testing Accuracy: %.2f%%" % (scores[1]*100))
	print("############################")
