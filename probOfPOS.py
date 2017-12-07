import nltk
import math
from nltk import ngrams

from nltk.tag.stanford import StanfordPOSTagger
st = StanfordPOSTagger('/home/saurav/Documents/postagger/models/english-bidirectional-distsim.tagger',
    '/home/saurav/Documents/postagger/stanford-postagger.jar')

posTagList =  ['NN', 'CC', 'LS', 'PDT', 'POS', 'SYM', 'NNS', 'NNP', 'NNPS', 'FW', 'CD', 'JJ', 'JJR', 'JJS', 'IN', 'TO', 'DT',
'EX', 'PRP', 'PRP$', 'WDT', 'WP', 'WP$', 'MD', 'VB', 'VBZ', 'VBP', 'VBD', 'VBN', 'VBG', 'RB', 'RBR', 'RBS', 'RP', 'WRB', 'UH', '.'] 

def blogCorpusPOS(brownCorp):
    outfile = open('realBlogCorpusPOS.txt', 'w')

    for sentence in brownCorp:
        tagSentence = ""
        sentence = sentence.decode('utf-8')
        tokensWord = nltk.word_tokenize(sentence)
        textToken = nltk.Text(tokensWord)
        #tags = st.tag(textToken)
        tags = nltk.pos_tag(tokensWord)

        for a,b in tags:
            if b in posTagList:
                tagSentence = tagSentence + b + " "

        tagSentence = tagSentence + "\n"

        outfile.write(tagSentence)

    outfile.close()

def calc_probabilities(brown):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    fourgram_p = {}
    fivegram_p = {}

    unigram = {}
    bigram = {}
    trigram = {}
    fourgram = {}
    fivegram = {}
    uni_count = biCount = triCount = fourCount = fiveCount = 0

    for sentence in brown:
        tokens = sentence.split()
    
        for word in tokens:
            uni_count += 1
            
            if word in unigram:
                unigram[word] += 1
            else:
                unigram[word] = 1

        bigram_tuples = tuple(nltk.bigrams(tokens))
        for item in bigram_tuples:
            biCount += 1
            if item in bigram:
                bigram[item] += 1
            else:
                bigram[item] = 1


        trigram_tuples = tuple(nltk.trigrams(tokens))        
        for item in trigram_tuples:
            triCount += 1
            if item in trigram:
                trigram[item] += 1
            else:
                trigram[item] = 1


        fourgram_tuples = ngrams(tokens, 4)        
        for item in fourgram_tuples:
            fourCount += 1
            if item in fourgram:
                fourgram[item] += 1
            else:
                fourgram[item] = 1

        fivegram_tuples = ngrams(tokens, 5)        
        for item in fivegram_tuples:
            fiveCount += 1
            if item in fivegram:
                fivegram[item] += 1
            else:
                fivegram[item] = 1    
                     

    # calculate unigram probability
    for word in unigram:
        temp = [word]
        unigram_p[tuple(temp)] = (float(unigram[word])/uni_count)

    # calculate bigram probability
    for word in bigram:
        bigram_p[tuple(word)] = (float(bigram[word])/biCount)

    # calculate trigram probability
    for word in trigram:
        trigram_p[tuple(word)] = (float(trigram[word])/triCount)

    # calculate fourgram probability
    for word in fourgram:
        fourgram_p[tuple(word)] = (float(fourgram[word])/fourCount) 

    # calculate fivegram probability
    for word in fivegram:
        fivegram_p[tuple(word)] = (float(fivegram[word])/fiveCount)  


    return unigram_p, bigram_p, trigram_p,fourgram_p,fivegram_p

def q1_output(unigrams, bigrams, trigrams,fourgrams ,fivegrams):
    #output probabilities
    outfile = open('probabilities.txt', 'w')
    for unigram in unigrams:
        outfile.write(unigram[0] + ':' + str(unigrams[unigram]) + '\n')
    for bigram in bigrams:
        outfile.write(bigram[0] + ' ' + bigram[1]  + ':' + str(bigrams[bigram]) + '\n')
    for trigram in trigrams:
        outfile.write(trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ':' + str(trigrams[trigram]) + '\n')

    for fourgram in fourgrams:
        outfile.write(fourgram[0] + ' ' + fourgram[1] + ' ' + fourgram[2] + ' ' + fourgram[3] + ':' + str(fourgrams[fourgram]) + '\n')

    for fivegram in fivegrams:
        outfile.write(fivegram[0] + ' ' + fivegram[1] + ' ' + fivegram[2] + ' ' + fivegram[3] + ' ' + fivegram[4]+ ':' + str(fivegrams[fivegram]) + '\n')

    outfile.close()

def main():
    infile = open('realBlogCorpus.txt', 'r')
    brown = infile.readlines()
    
    infile.close()

    blogCorpusPOS(brown)

    infile = open('realBlogCorpusPOS.txt','r')
    brownPOS = infile.readlines()
    infile.close()

    (a,b,c,d,e) = calc_probabilities(brownPOS)

    q1_output(a,b,c,d,e)

if __name__ == '__main__':
    main()
