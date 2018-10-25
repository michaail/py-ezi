import nltk
import string
import os
import numpy as np
import sys

# nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity
# from nltk.corpus import stopwords

token_dict = {}
stemmer = PorterStemmer()
# stopWords = stopwords.words('english')
count_cosine = True
d = False


# DOCUMENTS
documents = []
documents_title = []
with open('./documents/dTest.txt', 'r', encoding='iso-8859-1') as documents_file:
  temp = documents_file.read().split('\n\n')
  for doc in temp:
    # title = doc.split('\n', 1)[0]
    # documents_title.append(title)
    tmp = doc.replace('\n', ' ').replace('\n ', ' ').translate(str.maketrans("", "", string.punctuation)).lower()
    documents.append(tmp)

def stem_tokens(tokens, stemmer):
  stemmed = []
  for item in tokens:
    stemmed.append(stemmer.stem(item))
  return stemmed

def tokenize(text):
  tokens = nltk.word_tokenize(text)
  stems = stem_tokens(tokens, stemmer)
  return stems

def delete_multiple_occ(collection):
  for element in collection:
    if collection.count(element) > 1:
      print(element)
      collection.remove(element)

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=None, use_idf=True)
tfs = tfidf.fit_transform(documents)
idfs = tfidf.idf_

feature_names = tfidf.get_feature_names()

idf_dict = dict(zip(feature_names, idfs))

print(tokenize(documents[0]))

print(tfidf.vocabulary_)

print(idfs)

print(tfs)