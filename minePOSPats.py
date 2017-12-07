import nltk
from collections import Counter
import timeit

from nltk.tag.stanford import StanfordPOSTagger
st = StanfordPOSTagger('/home/saurav/Documents/postagger/models/english-bidirectional-distsim.tagger',
	'/home/saurav/Documents/postagger/stanford-postagger.jar')


Prob = {}
infile = open('probabilities.txt', 'r')
text = infile.readlines()

for sentence in text:
	keyValPair = sentence.split(":")
	Prob[keyValPair[0]] = float(keyValPair[1][:-1])

infile.close()


def prob(sequence):
	if sequence in Prob.keys():
		return Prob[sequence]
	else:
		return 0

def fairSCP(sequence):
	numerator = prob(sequence)*prob(sequence)
	sequence = sequence.split()

	denominator = 0

	for j in range(1,len(sequence)):
		seq1 = ""
		seq2 = ""
		cnt = 1
		
		for tag in sequence:
			if cnt <= j:
				seq1 = seq1 + tag + " "
				cnt += 1
			else:
				seq2 = seq2 + tag + " "


		seq2 = seq2[:-1]
		seq1 = seq1[:-1]
		
		denominator += prob(seq1)*prob(seq2)
		
	
	denominator = denominator*1.0/(len(sequence)-1)
	
	if denominator == 0:
		return 0.0

	SCP = numerator*1.0/denominator

	return SCP

minSup = 0.3
minAdherence = 0.2
tagList =  ['NN', 'CC', 'LS', 'PDT', 'POS', 'SYM', 'NNS', 'NNP', 'NNPS', 'FW', 'CD', 'JJ', 'JJR', 'JJS', 'IN', 'TO', 'DT',
'EX', 'PRP', 'PRP$', 'WDT', 'WP', 'WP$', 'MD', 'VB', 'VBZ', 'VBP', 'VBD', 'VBN', 'VBG', 'RB', 'RBR', 'RBS', 'RP', 'WRB', 'UH', '.'] 


infile = open('realBlogCorpusPOS.txt', 'r')
Doc = infile.readlines()
infile.close()


def candidateGen(Fk):
	Ck = []

	for item in Fk:
		for tag in tagList:
			itemTemp = item + " "+ tag
			Ck.append(itemTemp)


	return Ck


def minePOSPats(Doc):
	C = [{} for i in range(5)]
	F = [[] for i in range(5)]
	SP = [[] for i in range(5)]
	Cand = [[] for i in range(5)]

	n = len(Doc)	

	for post in Doc:
		for tag in tagList:
			if tag in post:
				if tag in C[0].keys():
					C[0][tag] += 1
				else:
					C[0][tag] = 1


	for a in C[0]:
		if C[0][a]*1.0/n >= minSup:
			F[0].append(a)
	
	SP[0] = F[0]
	temp={}
	for k in range(1,5):
		Cand[k] = candidateGen(F[k-1])
		for post in Doc:
			for candidate in Cand[k]:
				if candidate in post:
					if candidate in C[k].keys():
						C[k][candidate] += 1
					else:
						C[k][candidate] = 1

		for a in C[k]:
			if C[k][a]*1.0/n >= minSup:
				F[k].append(a)


		for a in F[k]:
			if fairSCP(a) >= minAdherence:
				SP[k].append(a)


	SPFinal = []
	SPFinal = SP[0]+SP[1]+SP[2]+SP[3]+SP[4]

	return SPFinal

#print(minePOSPats(Doc))
print("Mining POS sequence patterns...")
start_time = timeit.default_timer()
posFeatures = minePOSPats(Doc)
#print posFeatures

elapsed_time = timeit.default_timer() - start_time
print("Finished mining after %f seconds. " % (elapsed_time))

def POSFeatures(text):
	
	tokens = nltk.word_tokenize(text)
	text = nltk.Text(tokens)
	#tags = st.tag(text)
	tags = nltk.pos_tag(tokens)

	textTags = ""
	for word,tag in tags:
		if tag in tagList:
			textTags = textTags + tag + " "

	featureValues = []

	for feature in posFeatures:
		if feature in textTags:
			featureValues.append(1)
		else:
			featureValues.append(0)

	return tuple(featureValues)
