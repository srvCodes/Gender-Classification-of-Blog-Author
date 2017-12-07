import nltk
from collections import Counter
import string
import re
import yuleK
import sentiWordNet

from nltk.tag.stanford import StanfordPOSTagger
st = StanfordPOSTagger('/home/saurav/Documents/postagger/models/english-bidirectional-distsim.tagger',
	'/home/saurav/Documents/postagger/stanford-postagger.jar')

def baseFeatures(text):
	sentence_tokens = text.split('.')+text.split('?')+text.split('!')
	countSentences = len(sentence_tokens)-3

	if countSentences <= 0:
		countSentences = 1

	words = text.split()
	countWords = len(words)

	if countWords is not 0:
		countWordPerSentence = countWords*1.0/countSentences

		countCharacters = len(text)-countSentences

		countCharactersPerSentence = countCharacters*1.0/countSentences

		countAlphabets = sum(c.isalpha() for c in text)
		normalizedAlphabets = countAlphabets*1.0/countCharacters

		countDigits = sum(c.isdigit() for c in text)
		normalizedDigits = countDigits*1.0/countCharacters
	
		countSpaces  = sum(c.isspace() for c in text)
		normalizedSpaces = countSpaces*1.0/countCharacters

		countSpecialChars = countCharacters - countAlphabets - countDigits - countSpaces
		normalizedSpecialChars = countSpecialChars*1.0/countCharacters

		# we assumed short words are those words with length less than 4 characters
		countShortWords =  sum(1 for word in words if len(word) <= 4)
		normalizedShortWords = countShortWords*1.0/countWords

		countPunctuations = text.count('.')+text.count(',')+text.count('!')+text.count('?')+text.count(':')+text.count(';')
		doubleQuotes=re.findall(r'\"(.+?)\"',text)
		singleQuotes=re.findall(r'\'(.+?)\'',text)
		countPunctuations += len(singleQuotes) + len(doubleQuotes)
		normalizedPunctuations = countPunctuations*1.0/countCharacters

		averageWordLength = sum(len(word) for word in words)/len(words)

		countQuestionMark = text.count('?')
		if countPunctuations <= 0:
			countPunctuations = 1
		normalizedQuestionPerPunctuations = countQuestionMark*1.0/countPunctuations
		try:
			lexicalRichness = yuleK.yule(text)
		except:
			lexicalRichness = 0
		
		sentimentPosScore, sentimentNegScore = sentiWordNet.sentimentFeature(text)

		return (countSentences, countWords, countWordPerSentence, countCharacters, countCharactersPerSentence,
		normalizedAlphabets, normalizedDigits, normalizedSpaces, normalizedSpecialChars, normalizedShortWords,
		normalizedPunctuations, averageWordLength, normalizedQuestionPerPunctuations, lexicalRichness, sentimentPosScore,
		sentimentNegScore)

	else:
		return(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

#print(baseFeatures(""""""))
