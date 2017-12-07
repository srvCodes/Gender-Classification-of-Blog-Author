import os
import csv
import random

maleBlogs = []
femaleBlogs = []

# Adding more corpus
infile = open('maleBlogCorpus.txt', 'r')
add_male = infile.readlines()
maleBlogs = maleBlogs + add_male
infile.close()
infile = open('femaleBlogCorpus.txt', 'r')
add_female = infile.readlines()
femaleBlogs = femaleBlogs + add_female
infile.close()

labeled_male_blogs = ([(post[:-1], '0') for post in maleBlogs])
labeled_female_blogs = ([(post[:-1], '1') for post in femaleBlogs])

labeled_blogs = labeled_female_blogs+labeled_male_blogs
print(len(labeled_blogs))
print(labeled_blogs[1])

random.shuffle(labeled_blogs)
#train_set, test_set = labeled_blogs[2200:], labeled_blogs[:2200]

dest = "original_blogs.csv"

def write_to_csv():
	with open(dest, 'w', newline="") as out_file:
		writer = csv.writer(out_file, delimiter = '\t')
		writer.writerow(['Blog', 'Gender'])
		for row in labeled_blogs:
			writer.writerow(row)

write_to_csv()
	

