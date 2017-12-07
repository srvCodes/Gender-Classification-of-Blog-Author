import nltk
from collections import Counter

suffix_fdist = nltk.FreqDist()

def genderPreferentialFeatures(text):
	
	totalWords = len(text.split())

	text = nltk.word_tokenize(text)

	countSorry = 0

	for word in text:
		word = word.lower()
		suffix_fdist[word[-2:]] += 1
		suffix_fdist[word[-3:]] += 1
		suffix_fdist[word[-4:]] += 1
		if word.endswith('sorry'):
			countSorry += 1;


	return (suffix_fdist['able'], suffix_fdist['al'], suffix_fdist['ful'], suffix_fdist['ible'], suffix_fdist['ic'], suffix_fdist["'ll"], suffix_fdist["n't"], suffix_fdist["'d"],
 suffix_fdist["'re"], suffix_fdist["'ve"], suffix_fdist['ive'], suffix_fdist['less'], suffix_fdist['ly'], suffix_fdist['ous'], countSorry)
#http://kanagawa.lti.cs.cmu.edu/amls09/sites/default/files/argamon03.pdf (pg.13/26)


text = """This is hopeless countless priceless sorry and I am indecisive. so sorry I am feeling terrible 
that I am to fulfil a wonderful mathematical brutal vicious terrific problem."""
# print(genderPreferentialFeatures(text))
