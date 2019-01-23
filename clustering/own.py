import nltk
import string
import os
import numpy as np
import sys

import tfidf

from argparse import ArgumentParser
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity

token_dict = {}
stemmer = PorterStemmer()
count_cosine = True
d = False

documents_raw = []
documents = []
documents_title = []

# DOCUMENTS
with open('./data/documents-2.txt', 'r', encoding='iso-8859-1') as documents_file:
  temp = documents_file.read().split('\n\n')
  documents_raw = temp
  for doc in temp:
    title = doc.split('\n', 1)[0]
    documents_title.append(title)
    tmp = doc.replace('\n', ' ').replace('\n ', ' ').translate(str.maketrans("", "", string.punctuation)).lower()
    documents.append(tmp)


# KEYWORDS
keywords = ''
with open('./data/keywords-2.txt', 'r') as keywords_file:
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
  normalized = []
  for doc in bag:
    norm = dict.fromkeys(bag[0].keys(), 0)
    for term in doc:
      if max(doc.values()):
        norm[term] = doc[term] / max(doc.values())
    normalized.append(norm)
  return normalized

def get_values(doc):
  vector = []
  for word in doc:
    vector.append(doc[word])
  # vect = np.array(vector)
  return vector 

expand_query = False


# TOKENIZE
t_documents = []
t_keywords = tokenize(keywords)

delete_multiple_occ(t_keywords)

for doc in documents:
  t_documents.append(tokenize(doc))

# BAG OF WORDS
bw_documents = bag_of_words(t_documents, t_keywords)
norm_documents = normalize_bw(bw_documents)   # tf of documents

idfs = tfidf.idf(norm_documents, t_keywords)

d_tf_idf_vectors = []
for doc in norm_documents:
  tf_idf_d = tfidf.tfidf(doc, idfs)               # TF-IDF of document
  d_tf_idf_vectors.append(get_values(tf_idf_d))

def similarity(vector, matrix):
  length = np.linalg.norm(vector) * np.linalg.norm(matrix, axis=1)
  length[length == 0] = 1
  sim = np.sum(vector * matrix, axis=1) / length
  return np.argsort(sim)[::-1], sim


def clustering(words_matrix, k=9, max_iter=5):
  items_num = words_matrix.shape[0]
  assignment = np.empty(items_num)

  centroids = np.random.choice(items_num, size=k, replace=False)
  centroids = words_matrix[centroids, :]

  iter = 0
  while iter < max_iter:
    iter += 1
    changed = False

    for row_id, row in enumerate(words_matrix):
      sim, _ = similarity(row, centroids)
      cluster_id = sim[0]

      if assignment[row_id] != cluster_id:
        changed = True
        assignment[row_id] = cluster_id
    if not changed:
      break
    # calculate new centroids
    for cluster_id in range(k):
      indexes = np.argwhere(assignment == cluster_id)
      centroids[cluster_id] = np.mean(words_matrix[indexes, :], axis=0)
  return assignment, iter



def main():
  arr = np.array([np.array(xi) for xi in d_tf_idf_vectors])

  assignment, iterations = clustering(arr)

  documents_array = np.array(documents_raw)
  print(f'Clustering completed in {iterations} iterations')
  for cluster_id in range(9):
    indexes = np.argwhere(assignment == cluster_id)
    print(f'Cluster #{cluster_id}:')
    for row in documents_array[indexes]:
      title = row[0].split('\n')
      print(f"\t{title}")


main()