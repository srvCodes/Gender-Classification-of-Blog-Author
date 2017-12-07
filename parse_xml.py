# -*- coding: utf-8 -*-
import sys
import re 
import os
from bs4 import BeautifulSoup as Soup

dest1 = "/home/saurav/Documents/gender verification/jan31/new_male_corpus.txt"
dest2 = "/home/saurav/Documents/gender verification/jan31/new_female_corpus.txt"

#destn = "labelled_data.csv"
def parseLog(file, dest, gender):
    with open(file, 'rb') as handler:
        soup = Soup(handler, "html.parser")
        for message in soup.findAll('post'):
            #print(len(str(message).strip()))
            content = message.get_text()
            if(len(str(content).strip()) > 400):
                content = content.strip()
                re.sub("[^a-zA-Z0-9]", "", content)
                #content = inString.replace("\n", "")
                with open(dest, 'a', encoding="utf-8") as f:
                    f.write(content + "\t\t\t" + gender + "\n")

path = "/home/saurav/Documents/gender verification/jan31/blogs"

#cnt = 0
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
        filepath = os.path.join(path, file)
        filename = file.strip()
        filename = filename.split('.')
        gender = filename[1].strip()
        if(gender == "female"):
            parseLog(filepath, dest2, "female")
        else:
            parseLog(filepath, dest1, "male")
                   
#print(cnt)
