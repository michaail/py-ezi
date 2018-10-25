import nltk
import string
import os
import numpy as np
import sys

import tfidf
import rfeedback

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity

token_dict = {}
stemmer = PorterStemmer()
# stopWords = stopwords.words('english')
count_cosine = True
d = False

documents = []
documents_title = []


# DOCUMENTS
with open('./lab2/documents/dTest.txt', 'r', encoding='iso-8859-1') as documents_file:
  temp = documents_file.read().split('\n\n')
  for doc in temp:
    # title = doc.split('\n', 1)[0]
    # documents_title.append(title)
    tmp = doc.replace('\n', ' ').replace('\n ', ' ').translate(str.maketrans("", "", string.punctuation)).lower()
    documents.append(tmp)


# KEYWORDS
keywords = ''
with open('./lab2/documents/keywords.txt', 'r') as keywords_file:
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
      collection.remove(element)

def bag_of_words(collection, vocabulary):
  bag = []
  for doc in collection:
    doc_dict = {}
    for term in vocabulary:
      doc_dict[term] = 0
      if term in doc:
        for word in doc:
          if word == term:
            doc_dict[term] += 1
    bag.append(doc_dict)
  return bag

def normalize_bw(bag):
  for doc in bag:
    for term in doc:
      if max(doc.values()):
        doc[term] = doc[term] / max(doc.values())

def get_values(doc):
  vector = []
  for word in doc:
    vector.append(doc[word])
  # vect = np.array(vector)
  return vector 


query = 'information retrieval'


# TOKENIZE
t_documents = []
t_query = tokenize(query)
t_keywords = tokenize(keywords)

delete_multiple_occ(t_keywords)

for doc in documents:
  t_documents.append(tokenize(doc))


# BAG OF WORDS
bw_documents = bag_of_words(t_documents, tokenize('information retrieval agency'))
bw_query = bag_of_words([t_query], tokenize('information retrieval agency'))
# bw_documents = bag_of_words(t_documents, t_keywords)
# bw_query = bag_of_words(t_query, t_keywords)
normalize_bw(bw_documents)  # tf of documents
normalize_bw(bw_query)         # tf of query

idfs = tfidf.idf(bw_documents, tokenize('information retrieval agency'))
# tf_idfs_d = tfidf.tfidf(bw_documents[0], idfs)
tf_idf_q = tfidf.tfidf(bw_query[0], idfs)

# query_module = count_module(tf_idf_q)
# documents_modules = count_module(tf_idfs_d)

q_tf_idf_vector = get_values(tf_idf_q)
d_tf_idf_vectors = []
for doc in bw_documents:
  tf_idf_d = tfidf.tfidf(doc, idfs)               # TF-IDF of document
  # d_tf_idf_module.append(count_module(tf_idf_d))  # module of TF-IDF
  d_tf_idf_vectors.append(get_values(tf_idf_d))

# print(d_tf_idf_vectors)
# print(q_tf_idf_vector)
cosine_prototype = []
for vector in d_tf_idf_vectors:
  cosine_prototype.append(cosine_similarity([q_tf_idf_vector], [vector]))

cosine = []
for value in cosine_prototype:
  cosine.append(value[0][0])

print(cosine)

# print(cosine_similarity(tf_idfs_q.values(), tf_idfs_d.values()))

# print(tf_idfs_d)
# print(count_module(tf_idfs_d))



# print(bw_documents)