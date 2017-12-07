from sklearn.feature_extraction.text import CountVectorizer

def getCBOWfeatures(text):
	#initialize CountVectorizer object, which is scikit-learn's BOW tool
	#size of vocabulary = most frequent 5000 words occurring in whole document
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 400)

	#fit tranform does : a. learn the vocab by fitting to model b. transform train data into feature vectors
	train_data_features = vectorizer.fit_transform(text)

	#convert the result to numpy array since they are easy to work with	
	train_data_features = train_data_features.toarray()

	#print(train_data_features.shape)

	#words of the vocab
	#vocab = vectorizer.get_feature_names()
	#print(vocab) 

	#sum up the counts of each vocabulary word
	#dist = np.sum(train_data_features, axis = 0)

	#for each, print the word and the count
	#for tag, count in zip(vocab, dist):
	#	print(count, tag)

	return train_data_features