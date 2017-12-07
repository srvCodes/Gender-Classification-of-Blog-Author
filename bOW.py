import pandas as pd 
import numpy as np 
import nltk
import random
import csv
import re
import time #used to set start and end time while clustering
import logging #built-in logging module which is to be configured so that
			   #Word2Vec creates nice output messages

from bs4 import BeautifulSoup
from nltk.corpus import stopwords # for removal of stop words
from sklearn.feature_extraction.text import CountVectorizer #for creating Bag Of Words
import numpy as np 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier		
from sklearn import cross_validation
from sklearn import model_selection
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

#Conversion of train data to csv file
'''
infile = open('maleBlogCorpus.txt', 'r')
maleBlogs = infile.readlines()
infile.close()

infile = open('femaleBlogCorpus.txt', 'r')
femaleBlogs = infile.readlines()
infile.close()

#label 0 for female, 1 for male
labeled_male_blogs = ([(post[:-1], '1') for post in maleBlogs])
labeled_female_blogs = ([(post[:-1], '0') for post in femaleBlogs])

labeled_blogs = labeled_male_blogs + labeled_female_blogs
random.shuffle(labeled_blogs)

print(labeled_blogs[100])

with open('labelledTrainData.csv', 'w') as out:
	writer = csv.writer(out, delimiter = '\t')
	writer.writerow(['blogs', 'label'])
	writer.writerows(labeled_blogs[:2440])

with open('labelledTestData.csv', 'w') as out:
	writer = csv.writer(out, delimiter = '\t')
	writer.writerow(['blogs', 'label'])
	writer.writerows(labeled_blogs[2441:])
'''
train = pd.read_csv("labelledTrainData.csv", header = 0, delimiter = "\t", quoting = 3)
test = pd.read_csv("labelledTestData.csv", header = 0, delimiter = "\t", quoting=3)

#print(train.shape)
#print(train["blogs"][0])

#function returning preprocessed data from raw data
def review_to_words(raw_review):
	#remove html tags
	review_text = BeautifulSoup(raw_review, "html.parser").get_text()

	#remove non-letters
	letters_only = re.sub("[^a-zA-Z]", " ", review_text)

	#convert to lower case, split into individuall words
	words = letters_only.lower().split()

	#convert to set coz in python, searching of set is much faster than searching a list
	stops = set(stopwords.words("english"))

	#remove stop words
	meaningful_words = [w for w in words if not w in stops]

	#join the words back into one string separated by space and return result
	return(" ".join(meaningful_words))

#print(review_to_words(train["blogs"][0]))

#size of the data frame 
num_reviews = train["blogs"].size

#initialize an empty list to hold the clean reviews
clean_train_reviews = []
stemmed_reviews = []
lemmatized_reviews = []

#loop over each review and append it to clean_train_review
print("cleaning and parsing the text data..")

for i in range(0, num_reviews):
	#print status in every 800 reviews
	if((i+1) % 800 == 0):
		print("Review %d of %d\n" % (i+1, num_reviews))
	clean_train_reviews.append(review_to_words(train["blogs"][i]))

#print(clean_train_reviews[0])

num = len(clean_train_reviews)
lmtzr = nltk.stem.wordnet.WordNetLemmatizer()

print("Lemmatizing the preprocessed data.. ")
for i in range(0, num):
	if((i+1) % 800 == 0):
		print("Review %d of %d\n" % (i+1, num))
	res = " ".join([ lmtzr.lemmatize(kw) for kw in clean_train_reviews[i].split(" ")])
	lemmatized_reviews.append(res)

stemmer = nltk.PorterStemmer()
num = len(lemmatized_reviews)

print("Stemming the lemmatized data.. ")
for i in range(0, num):
	if((i+1) % 800 == 0):
		print("Review %d of %d\n" % (i+1, num))
	res = " ".join([ stemmer.stem(kw) for kw in lemmatized_reviews[i].split(" ")])
	stemmed_reviews.append(res)
'''
#print(stemmed_reviews[1])

#initialize CountVectorizer object, which is scikit-learn's BOW tool
#size of vocabulary = most frequent 5000 words occurring in whole document
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 400)

#fit tranform does : a. learn the vocab by fitting to model b. transform train data into feature vectors
train_data_features = vectorizer.fit_transform(stemmed_reviews)

#convert the result to numpy array since they are easy to work with
train_data_features = train_data_features.toarray()

#print(train_data_features.shape)

print(train_data_features[1])


#words of the vocab
vocab = vectorizer.get_feature_names()
#print(vocab) 

#sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis = 0)

#for each, print the word and the count
#for tag, count in zip(vocab, dist):
#	print(count, tag)

###############Classification#################
#forest =RandomForestClassifier(n_estimators = 20)

#forest = forest.fit(train_data_features, train["label"])

##################Feature extraction of test data#################
print(test.shape)
'''
numTestreviews = len(test["blogs"])

clean_test_reviews = []
stemmed_test_reviews = []
lemmatized_test_reviews = []

print("cleaning and parsing the test data..")

for i in range(0, numTestreviews):
	#print status in every 800 reviews
	if((i+1) % 800 == 0):
		print("Review %d of %d\n" % (i+1, numTestreviews))
	clean_test_reviews.append(review_to_words(test["blogs"][i]))

#print(clean_train_reviews[0])

num = len(clean_test_reviews)
lmtzr = nltk.stem.wordnet.WordNetLemmatizer()

print("Lemmatizing the preprocessed data.. ")
for i in range(0, num):
	if((i+1) % 800 == 0):
		print("Review %d of %d\n" % (i+1, num))
	res = " ".join([ lmtzr.lemmatize(kw) for kw in clean_test_reviews[i].split(" ")])
	lemmatized_test_reviews.append(res)

stemmer = nltk.PorterStemmer()
num = len(lemmatized_test_reviews)

print("Stemming the lemmatized data.. ")
for i in range(0, num):
	if((i+1) % 800 == 0):
		print("Review %d of %d\n" % (i+1, num))
	res = " ".join([ stemmer.stem(kw) for kw in lemmatized_test_reviews[i].split(" ")])
	stemmed_test_reviews.append(res)
'''
test_data_features = vectorizer.transform(stemmed_test_reviews)
test_data_features = test_data_features.toarray()

tr = train["label"].tolist()
te = test["label"].tolist()
mylist = tr + te 
print(len(mylist))
#result = forest.predict(test_data_features)
#scores = model_selection.cross_val_score(forest, test_data_features, test["label"], cv = 10)

#avgScore = sum(scores)/len(scores)
#print(avgScore) '''

def blog_to_wordlist(blogs, remove_stopwords = False):
	#function to convert a blog to a sequence of words, optionally 
	#removing stopwords; returns a list of words
	review_text = BeautifulSoup(blogs, "html.parser").get_text()

	review_text = re.sub("[^a-zA-Z0-9]", " ", review_text)

	words = review_text.lower().split()

	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words = [w for w in words if not w in stops]

	return(words)

#we want an input format that is list of sentences with each list = a list of words
#use NLTK's punkt tokenizer for sentence splitting
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def blogs_to_sentences(blog, tokenizer, remove_stopwords = False):
	#function to split a blog into parsed sentences. 
	# returns a list of sentences where each sentence = list of words

	#use NLTK tokenizer to split the paragraph into sentences
	raw_sentences = tokenizer.tokenize(blog.strip())

	#loop over each sentence
	sentences = []
	for raw_sentence in raw_sentences:
		if(len(raw_sentence) > 0):
			#call blog_to_wordlist to get a list of words
			sentences.append(blog_to_wordlist(raw_sentence))

	return sentences

sentences = []
print("Parsing sentences from train set\n")
for blog in train["blogs"]:
	sentences += blogs_to_sentences(blog, tokenizer, remove_stopwords = True)

print(len(sentences))

#print(sentences[1])
#print(sentences[2])
#print(sentences[3])

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s',
	level = logging.INFO)

#set values for parameters
num_features = 300 		#word vector dimensionality
min_word_count = 40 	#min word count for choosing that word
num_workers = 4			#no of threads to run in parallel
context = 10			#context window size
downsampling = 1e-3		#downsample setting for frequent words

print("Training model.. \n")
model = word2vec.Word2Vec(sentences, workers = num_workers, 
			size = num_features, min_count = min_word_count,
			window = context, sample = downsampling)

#if no more training of model require, below will make the model much
#more memory-efficient
model.init_sims(replace=True)

#Creating a meaningful model name and saving it for later use by
# Word2Vec.load()
model_name = "bOW_Gen_Ver_1"
model.save(model_name)

#print(model.doesnt_match("man woman child kitchen".split()))

#print(model.doesnt_match("france england germany berlin".split()))

#print(model.doesnt_match("paris berlin london austria".split()))

#print(model.most_similar("man"))

#print(model.most_similar("queen"))

print(model.wv.syn0.shape)

# use "vector quantization" to group vector of words in clusters

start = time.time()

#set "k" = 1/5th of vocab size or an avg of 5 words per cluster
word_vectors = model.wv.syn0
print(word_vectors.shape[0])

num_clusters = int(word_vectors.shape[0] / 8)

#initialize a k-means object and use it to extract centroids
kmeans_clust = KMeans(n_clusters = num_clusters)
idx = kmeans_clust.fit_predict(word_vectors)

#get the end time and print how long the process took
end = time.time()
print("Time taken for K means clustering = ",(end - start), " seconds." )

#cluster assignment for each word now stored in idx
# map each vocab. word to a cluster number through zipping
word_centroid_map = dict(zip(model.wv.index2word, idx))

# fucntion to convert blogs into bag-of-centroids
# is just like bag-of-words but uses semantically related clusters instead
# of individual words; return a numpy array for each blog
def create_bag_of_centroids(wordlist, word_centroid_map):
	# no of clusters equal to the highest cluster index in the word-centroid map
	num_centroids = max(word_centroid_map.values()) + 1

	#pre-allocate the bag of centroid vectors (for speed)
	bag_of_centroids = np.zeros(num_centroids, dtype = "float32")

	#loop over the words in the blog; if word is in the vocab,
	# find which cluster it belongs to, and increment that cluster count by one
	for word in wordlist:
		if word in word_centroid_map:
			index = word_centroid_map[word]
			bag_of_centroids[index] += 1 

	#return bag-of-centroids, a np array
	return bag_of_centroids

# pre-allocate an array for training set bags of centroids (for speed)
train_centroids = np.zeros((train["blogs"].size, num_clusters), dtype= "float32")

# transform training set blogs into bags of centroids
counter= 0
for blog in stemmed_reviews: 
	train_centroids[counter] = create_bag_of_centroids(blog, word_centroid_map)
	counter += 1

test_centroids = np.zeros((test["blogs"].size, num_clusters), dtype="float32")

counter = 0
for blog in stemmed_test_reviews:
	test_centroids[counter] = create_bag_of_centroids(blog, word_centroid_map)
	counter += 1

alg = RandomForestClassifier()

print("Fitting a RandomForestClassifier to labelled train data.. ")
alg = alg.fit(train_centroids, train["label"])

y_pred = alg.predict_proba(test_centroids)[:,1]
fpr, tpr, thresholds = roc_curve(test["label"], y_pred)
print("AUC_ROC for Voting classifier: %f" % auc(fpr, tpr))



