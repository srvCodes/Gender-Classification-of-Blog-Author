import nltk

infile = open('sentiWordNet.txt', 'r')
swnLines = infile.readlines()
infile.close()

swnPos = {}
swnNeg = {}
count = {}

i = 0
for item in swnLines:
    swnLines[i]=swnLines[i].split('\t')
    posScore = swnLines[i][2]
    negScore = swnLines[i][3]

    words = swnLines[i][4].split(' ')
    for word in words:
        j = word.find('#')
        word = word[:j]

        if word in count.keys():
            count[word] += 1
            swnPos[word] += float(posScore)
            swnNeg[word] += float(negScore)
        else:
            count[word] = 1
            swnPos[word] = float(posScore)
            swnNeg[word] = float(negScore)

    i += 1


for item in count:
    swnPos[item] = swnPos[item]*1.0/count[item]
    swnNeg[item] = swnNeg[item]*1.0/count[item]

def sentimentFeature(text):
    posSentiSum = 0
    negSentiSum = 0
    tokens = nltk.word_tokenize(text)
    noOfWords = len(text.split())
    for word in tokens:
        try:
            posScore = swnPos[word]
            negScore = swnNeg[word]
        except KeyError:
            posScore = 0.0
            negScore = 0.0

        posSentiSum += posScore
        negSentiSum += negScore

    posSenti = posSentiSum*1.0/noOfWords
    negSenti = negSentiSum*1.0/noOfWords

    return(round(posSenti,3), round(negSenti,3))


#print(sentimentFeature("""i hate abc"""))

#try:
#   print(swnNeg[':'])
#except KeyError:
#    print(0)