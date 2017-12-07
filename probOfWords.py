import nltk
import math
from nltk.corpus import stopwords
from nltk import ngrams
'''
def blogCorpusPOS(brownCorp):
    outfile = open('realBlogCorpusWords.txt', 'w')

    for sentence in brownCorp:
        
        totalWords = ""
        WordsTokenInOneSentence = nltk.word_tokenize(sentence)
        WordsInOneSentence = nltk.Text(WordsTokenInOneSentence)

        for word in WordsInOneSentence:
            for i in range(3,6):
                for j in range(len(word)-i+1):
                    totalWords = totalWords + word[j:j+i] + " " 

        totalWords = totalWords + "\n"

        outfile.write(totalWords)

    outfile.close()

def calc_count(brown):
    
    count = 0
    wordCount = {}
    wordCount_p = {}

    for sentence in brown:
        words = sentence.split()
    
        for word in words:
            count += 1
            
            if word in wordCount:
                wordCount[word] += 1
            else:
                wordCount[word] = 1

    # calculate probability
    #for word in wordCount:
    #    temp = [word]
    #    wordCount_p[tuple(temp)] = (float(wordCount[word])/count)

    return wordCount
'''

def q1_output(wordCount,filename):
    outfile = open(filename, 'w')
    for key,val in wordCount.items():
        if val >= 3000:
            outfile.write(key + ': ' + str(val) + '\n')
     
    outfile.close()

def main():
    with open('realBlogCorpus.txt', 'r') as content_file:
        content = content_file.read()
    content = content.lower()
    stops = set(stopwords.words('english'))

    for i in range(3,6):
        word_Count = {}
        totalcount = 0
        for j in range(len(content)-i+1):
            if content[j:j+i] in word_Count and content[j:j+i] not in stops:
                word_Count[content[j:j+i]] += 1
            else:
                word_Count[content[j:j+i]] = 1  
            totalcount += 1 
        filename = "count_Words" + str(i) + ".txt"
        print(totalcount)
        q1_output(word_Count,filename)

if __name__ == '__main__':
    main()
