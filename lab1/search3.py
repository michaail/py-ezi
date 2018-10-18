import nltk
import string
import os
import numpy as np
import sys

# nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
# from nltk.corpus import stopwords

token_dict = {}
stemmer = PorterStemmer()
# stopWords = stopwords.words('english')
count_cosine = True
d = False

# options
query_string = 'machine'
if len(sys.argv) == 2:
  if sys.argv[1][0] == '-':
    count_cosine = False
    print('options')
    if sys.argv[1][1] == 'd':
      d = True
      print('show preprocessed documents')
  else:
    count_cosine = True
    query_string = sys.argv[1]


# DOCUMENTS
documents = []
documents_title = []
with open('./documents/documents.txt', 'r', encoding='iso-8859-1') as documents_file:
  temp = documents_file.read().split('\n\n')
  for doc in temp:
    title = doc.split('\n', 1)[0]
    documents_title.append(title)
    tmp = doc.replace('\n', ' ').replace('\n ', ' ').translate(str.maketrans("", "", string.punctuation)).lower()
    documents.append(tmp)
  

# KEYWORDS
keywords = ''
with open('./documents/keywords.txt', 'r') as keywords_file:
  keywords = keywords_file.read().replace('\n', ' ').lower()

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

t_keywords = tokenize(keywords)

delete_multiple_occ(t_keywords)

print(len(t_keywords))

#this can take some time 
if count_cosine:
  tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=None, vocabulary=t_keywords)
  tfs = tfidf.fit_transform(documents)

  feature_names = tfidf.get_feature_names()

  # similarity_matrix = cosine_similarity(tfs)
  print('Asked querry:\t"' + query_string + '"')
  queryTFIDF = tfidf.transform([query_string])

  cosine = cosine_similarity(queryTFIDF, tfs).flatten()
  related_docs = cosine.argsort()[:-11:-1]

  for r_doc in related_docs:
    str_val = cosine[r_doc]
    print(documents_title[r_doc] + "\tscore: " + str(str_val))

if d:
  for docu in documents:
    stemmed = tokenize(docu)
    print('stemmed doc:\n\t' + '"' + ' '.join(stemmed) + '"' + '\n\n')
