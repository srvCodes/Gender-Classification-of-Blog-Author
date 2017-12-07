import nltk
import random

import FMeasure
import wordClassFeatures
import genderPreferentialFeatures
import minePOSPats
import baseFeatures
	
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC, SVR
from nltk.classify import ClassifierI
from sklearn.model_selection import GridSearchCV

def findFeature(text):
	fMeasureFeature = FMeasure.FMeasure(text)
	wordClassFeature = wordClassFeatures.wordClassFeatures(text)
	genderPreferentialFeature = genderPreferentialFeatures.genderPreferentialFeatures(text)
	posFeature = minePOSPats.POSFeatures(text)
	baseFeature = baseFeatures.baseFeatures(text)

	fMeasureToTuple = []
	featureVector = {}
	fMeasureToTuple.append(fMeasureFeature)

	features = tuple(fMeasureToTuple) + wordClassFeature + genderPreferentialFeature + baseFeature + posFeature
	cnt = -1
	for feature in features:
		cnt += 1
		featureVector[cnt] = feature

	return featureVector


infile = open('maleBlogCorpus.txt','r')
maleBlogs = infile.readlines()
infile.close()

infile = open('femaleBlogCorpus.txt','r')
femaleBlogs = infile.readlines()
infile.close()


labeled_male_blogs = ([(post, 'male') for post in maleBlogs])
labeled_female_blogs = ([(post, 'female') for post in femaleBlogs])

labeled_blogs = labeled_female_blogs+labeled_male_blogs
random.shuffle(labeled_blogs)

featuresets = [(findFeature(post), gender) for (post, gender) in labeled_blogs]

#featureimage = [[] for i in range(999)]

#cnt = 0

#for featureVec in featuresets:
#	for i in featureVec:
#		featureimage[cnt].append(featureVec[i])

#	cnt += 1


print (featuresets[1])

train_set, test_set = featuresets[500:], featuresets[:500]

'''LogisticRegression_classifier = SklearnClassifier(LogisticRegression( max_iter=200))
LogisticRegression_classifier.train(train_set)
print("LogisticRegression Accuracy: ", nltk.classify.accuracy(LogisticRegression_classifier, test_set))

#NBclassifier = nltk.NaiveBayesClassifier.train(train_set)
#print("NBClassifier Accuracy: ", nltk.classify.accuracy(NBclassifier, test_set))

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train_set)
print("BernoulliNB Accuracy: ", nltk.classify.accuracy(BernoulliNB_classifier, test_set))

LinearSVC_classifier = SklearnClassifier(LinearSVC(C=0.03, penalty='l1', dual=False))
LinearSVC_classifier.train(train_set)
print("LinearSVC_classifier Accuracy: ", nltk.classify.accuracy(LinearSVC_classifier, test_set))
'''
SVR_classifier = SklearnClassifier(SVR(kernel='rbf', gamma=2.0, C=10))
SVR_classifier.train(train_set)
print("SVR_regression Accuracy: ", nltk.classify.accuracy(SVR_classifier, test_set))

'''NuSVC_classifier = SklearnClassifier(NuSVC(cache_size=500, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
      max_iter=-1, nu=0.5, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False))
NuSVC_classifier.train(train_set)
print("NuSVC_classifier Accuracy: ", nltk.classify.accuracy(NuSVC_classifier, test_set))
#nu should be strictly less than 1, 0.9, 0.99, 0.999..
#dont use rbf kernels for texts'''
'''NuSVC(cache_size=500, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='poly',
      max_iter=-1, nu=0.7, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)'''

