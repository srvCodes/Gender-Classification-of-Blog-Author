# Gender Classification of Blog Authors

  This repository contains the entire source code for implementing our paper **Gender Classification of Blog Authors: With Feature Engineering and Deep Learning using LSTM Networks<sup>1</sup>**.
  
## Prerequisites:
  * nltk version 3.2.2
  * scikit-learn  0.18.1
  * Keras 2.0.6 (Tensorflow backend version: 1.0.1)
  
We used the data set originally mentioned by Mukherjee and Liu<sup>2</sup> in their work as well as [The Blog Authorship Corpus](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm) for showing our results.

## Feature Extraction:
The input fed to each of the module mentioned below is the blog post after minimal processing (removing stopwords and html tags):

### Newly Added Feature Classes:
- [mineCharPats.py](https://github.com/Saurav0074/gender_blog/blob/master/mineCharPats.py) mines the *Character Sequence Pattern Features*.
- [wordClassFeatures.py](https://github.com/Saurav0074/gender_blog/blob/master/wordClassFeatures.py) mines the word class factors along with the 13 new word classes proposed by us.
- [baseFeatures.py](https://github.com/Saurav0074/gender_blog/blob/master/baseFeatures.py) contains all the surface features used by us. These include: *Normalized count of sentences, Normalized count of words, Normalized count of characters, Normalized count of alphabets, Normalized count of digits, Normalized count of special characters,
Normalized count of punctuation marks, Count of short words (< 4 characters)* and *Average word length*.
- [sentiWordNet.py](https://github.com/Saurav0074/gender_blog/blob/master/sentiWordNet.py) measures the average sentiment score based on the Senti WordNet 3.0 Lexical Resource.
- [yuleK.py](https://github.com/Saurav0074/gender_blog/blob/master/yuleK.py) measures the lexical richness of the blog based on Yule's K index.

### Re-implemented Feature Classes:
- [minePOSPats.py](https://github.com/Saurav0074/gender_blog/blob/master/minePOSPats.py) mines the variable length POS sequence patterns on the basis of minimum support and minimum adherence thresholds specified by the user. Prior to running this file, the user needs to find the POS probability of all such words using [probOfPOS.py](https://github.com/Saurav0074/gender_blog/blob/master/probOfPOS.py).
- [FMeasure.py](https://github.com/Saurav0074/gender_blog/blob/master/FMeasure.py) measures the text’s relative contextuality
(implicitness), as opposed to the formality (explicitness).
- [genderPreferentialFeatures.py](https://github.com/Saurav0074/gender_blog/blob/master/genderPreferentialFeatures.py) gives a measure of 10 distinguishing word endings.
- [get_CBOW_features.py](https://github.com/Saurav0074/gender_blog/blob/master/get_CBOW_features.py) extracts the Continuous Bag Of Words from the text. *However, this didn't lead to any substantial improvement in the accuracy of the model.*

### Classification Algorithms
- [first_approach.py](https://github.com/Saurav0074/gender_blog/blob/master/first_approach.py) implements a Voting of Machine Learning Classifiers based on the features extracted above.

## References
[1] S. Jha, V. P. Dwivedi, D. K. Singh, and Ranvijay, “Gender classification of blog authors: With feature
engineering and deep learning using lstm networks,” in *Proceedings of the Ninth International Conference on Advanced
Computing (ICoAC-2018)*.

[2] A. Mukherjee and B. Liu, “Improving gender classification of blog authors,” in *EMNLP*, 2010.
