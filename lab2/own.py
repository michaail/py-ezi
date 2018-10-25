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

# options
query_string = 'papers'
if len(sys.argv) == 2:
  # # if sys.argv[1][0] == '-':
  # #   count_cosine = False
  # #   print('options')
  # #   if sys.argv[1][1] == 'd':
  # #     d = True
  # #     print('show preprocessed documents')
  # else:
  # count_cosine = True
  query_string = sys.argv[1]
elif len(sys.argv) == 3:
  if sys.argv[1] == '-e':
    expand_query = True
  query_string = sys.argv[2] 

# query_string = 'papers'
# query_string = sys.argv[1]

# TOKENIZE
t_documents = []
t_query = tokenize(query_string)
t_keywords = tokenize(keywords)

delete_multiple_occ(t_keywords)

for doc in documents:
  t_documents.append(tokenize(doc))

# BAG OF WORDS
# bw_documents = bag_of_words(t_documents, tokenize('information retrieval agency'))
# bw_query = bag_of_words([t_query], tokenize('information retrieval agency'))
bw_documents = bag_of_words(t_documents, t_keywords)
bw_query = bag_of_words([t_query], t_keywords)
norm_documents = normalize_bw(bw_documents)   # tf of documents

# idfs = tfidf.idf(norm_documents, tokenize('information retrieval agency'))
idfs = tfidf.idf(norm_documents, t_keywords)

d_tf_idf_vectors = []
for doc in norm_documents:
  tf_idf_d = tfidf.tfidf(doc, idfs)               # TF-IDF of document
  # d_tf_idf_module.append(count_module(tf_idf_d))  # module of TF-IDF
  d_tf_idf_vectors.append(get_values(tf_idf_d))



def ask_query(query, expand_query):
  norm_query = normalize_bw(query)           # tf of query_string
  # tf_idfs_d = tfidf.tfidf(norm_documents[0], idfs)
  tf_idf_q = tfidf.tfidf(norm_query[0], idfs)
  
  q_tf_idf_vector = get_values(tf_idf_q)

  cosine_prototype = []
  for vector in d_tf_idf_vectors:
    cosine_prototype.append(cosine_similarity([q_tf_idf_vector], [vector]))

  cosine = []
  for value in cosine_prototype:
    cosine.append(value[0][0])


  i = 1
  related_docs = np.array(cosine).argsort()[:-11:-1]
  for r_doc in related_docs:
    str_val = cosine[r_doc]
    print(str(i) + ". " + documents_title[r_doc] + "\tscore: " + str(str_val))
    i += 1

  cont = input("Czy chcesz rozszerzyc zapytanie (y/N)? ")
  if cont == 'Y' or cont == 'y':
    expand_query = True
  else:
    return

  if expand_query:
    rel = input("wybierz odpowiadajace dokumenty (oddziel spacja): ").split(' ')
    relevant = [int(numeric_string) for numeric_string in rel]
    print(relevant)

    not_relevant = []
    for index in range(1, 11):
      if index not in relevant:
        not_relevant.append(index)

    relevant_documents = []
    for r_doc in relevant:
      relevant_documents.append(bw_documents[related_docs[r_doc - 1]])

    not_relevant_documents = []
    for r_doc in not_relevant:
      not_relevant_documents.append(bw_documents[related_docs[r_doc - 1]])

    new_query = rfeedback.rocchio(bw_query, relevant_documents, not_relevant_documents, t_keywords)
    nz_n_query = {}
    for key, value in new_query.items():
      if value:
        nz_n_query[key] = value

    print("Nowe zapytanie: \n")
    print(nz_n_query)
    print("")


    res = input("Czy chcesz wykonac to zapytanie (y/N) ")
    if res == 'Y' or res == 'y':
      expand_query = False
      ask_query([new_query], expand_query)
    else:
      return
  else:
    cont = input("Czy chcesz rozszerzyc zapytanie (y/N)?")
    if cont == 'Y' or cont == 'y':
      expand_query = True
    else:
      return
  return
  # print(relevant_documents)

# print(cosine_similarity(tf_idfs_q.values(), tf_idfs_d.values()))

# print(tf_idfs_d)
# print(count_module(tf_idfs_d))

ask_query(bw_query, expand_query)

# print(bw_documents)