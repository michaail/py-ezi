# Compute TF vector for given document
def tf(wordDict, bow):
  tfDict = {}
  bowCount = len(bow)
  for word, count in wordDict.items():
    tfDict[word] = count/float(bowCount)
  return tfDict


def idf(docList, keywords):
  import math
  idfDict = {}
  N = len(docList)

  idfDict = dict.fromkeys(keywords, 0)
  for doc in docList:
    for word, val in doc.items():
      if val > 0:
        idfDict[word] += 1
  
  for word, val in idfDict.items():
    if val:
      idfDict[word] = math.log10(N / float(val))
      
  return idfDict


def tfidf(tfBow, idfs):
  tfidfDict = {}
  for word, val in tfBow.items():
    tfidfDict[word] = val*idfs[word]

  return tfidfDict
