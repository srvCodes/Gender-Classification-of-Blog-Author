import nltk
from collections import Counter
import timeit

def CharFeatures(text):
	print("Mining Words sequence patterns...")
	#start_time = timeit.default_timer()

	#elapsed_time = timeit.default_timer() - start_time
	#print("Finished mining after %f seconds. " % (elapsed_time))

	tags = []

	infile = open('count_Words3.txt', 'r')
	content = infile.readlines()
	infile.close()

	for sentence in content:
		sentence = sentence.split(':')
		tags.append(sentence[0])

	infile = open('count_Words4.txt', 'r')
	content = infile.readlines()
	infile.close()

	for sentence in content:
		sentence = sentence.split(':')
		tags.append(sentence[0])

	infile = open('count_Words5.txt', 'r')
	content = infile.readlines()
	infile.close()

	for sentence in content:
		sentence = sentence.split(':')
		tags.append(sentence[0])
	#print(tags)
		
	featureValues = []
	Words = []
	for i in range(3,6):
	    for j in range(len(text)-i+1):
	    	Words.append(text[j:j+i])

	for key in tags:
	    if key in Words:
	    	featureValues.append(1)
	    else:
	    	featureValues.append(0)

	#print(featureValues)
	return tuple(featureValues)
