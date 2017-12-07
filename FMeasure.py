from __future__ import division
import nltk
from collections import Counter

def FMeasure(text):
	#text = "Hey bro! Can I get your cool shoes?"
	#text = text.decode('utf-8')
	totalWords = len(text.split())

	if totalWords is not 0:
		tokens = nltk.word_tokenize(text)
		text = nltk.Text(tokens)
		#tags = st.tag(text)
		tags = nltk.pos_tag(tokens)

		posCounts = Counter(tag for word,tag in tags)

		countNoun = posCounts['NN']+posCounts['NNS']+posCounts['NNP']+posCounts['NNPS']+posCounts['FW']
		countAdj = posCounts['CD']+posCounts['JJ']+posCounts['JJR']+posCounts['JJS']
		countPrep = posCounts['IN']+posCounts['TO']
		countArt = posCounts['DT']
		countPron = posCounts['EX']+posCounts['PRP']+posCounts['PRP$']+posCounts['WDT']+posCounts['WP']+posCounts['WP$']
		countVerb = posCounts['MD']+posCounts['VB']+posCounts['VBZ']+posCounts['VBP']+posCounts['VBD']+posCounts['VBN']+posCounts['VBG']
		countAdverb = posCounts['RB']+posCounts['RBR']+posCounts['RBS']+posCounts['RP']+posCounts['WRB']
		countIntj = posCounts['UH']

		freqNoun = countNoun*100.0/totalWords
		freqAdj = countAdj*100.0/totalWords
		freqPrep = countPrep*100.0/totalWords
		freqArt = countArt*100.0/totalWords
		freqPron = countPron*100.0/totalWords
		freqVerb = countVerb*100.0/totalWords
		freqAdverb = countAdverb*100.0/totalWords
		freqIntj = countIntj*100.0/totalWords

		FMeasure = (freqNoun+freqAdj+freqPrep+freqArt-freqPron-freqVerb-freqAdverb-freqIntj+100)/2

		return FMeasure*1.0/100
	else:
		return 0.0

#print(FMeasure("Check if you have added the Stanford Parser path to CLASSPATH environment variable."))
