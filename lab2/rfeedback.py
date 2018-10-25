def rocchio(query, relevants, notRelevants, dictionary):
  alpha = 1
  beta = 0.75
  gamma = 0.25

  rel = {}
  nRel = {}

  for term in dictionary:
    rel[term] = 0
    nRel[term] = 0

  for doc in relevants:
    for term in doc:
      rel[term] += doc[term]

  for doc in notRelevants:
    for term in doc:
      nRel[term] += doc[term]

  # print(rel)
  # print(nRel)

  newQuery = {}

  for term in dictionary:
    tmp = 0
    if term in query:
      tmp += query[term] * alpha
    if term in rel:
      tmp += (rel[term] * beta) / len(relevants)
    if term in nRel:
      tmp -= (nRel[term] * gamma) / len(notRelevants)

    if tmp < 0:
      newQuery[term] = 0
    else:
      newQuery[term] = tmp

  # print(newQuery)
  return newQuery
  
# test only 
# rocchio({"cheap": 3, "CDs": 2, "DVDs": 1, "extremely": 1}, [{"cheap": 2, "CDs": 2, "software": 1}], [{"cheap": 1, "DVDs": 1, "thrills": 1}], ['cheap', 'CDs', 'DVDs', 'extremely', 'software', 'thrills'])

