import scipy.sparse
import numpy as np
import sys
from spectral_lda import spectral_lda

'''
A, B, C = np.loadtxt('../a.txt', skiprows = 0, unpack = True)
ij = np.vstack((A,B))
ji = np.vstack((B,A))
m1 = scipy.sparse.coo_matrix((C, ij))
m3 = m1.todense()
'''
numTopics = 15
vocabsize = 100
numDocuments = 500
# a1 = np.zeros((101,102660))
a1 = np.zeros((numDocuments, vocabsize))

inputFile = sys.argv[1]
f1 = open(inputFile,'r')

docId = 0

for line in f1:
	flds = line.strip().split()

	for j in range(len(flds)):
		a1[docId, j] = int(flds[j])

	docId += 1


alpha, beta = spectral_lda(a1, alpha0=1, k=numTopics, l1_simplex_proj=True)

f3 = open('beta.txt','w')
for i in range(len(beta)):
	# print(beta[i])
	f3.write(' '.join([str(j) for j in beta[i]]) + "\n")
f3.close()

# print(alpha)

betaTranspose = list(np.transpose(beta))

# evaluation

origTopicDistFilePtr = open("../topicWordDistributionSingleTopic.txt")
origTopicDist = []

for line in origTopicDistFilePtr:

	flds = line.strip().split()
	topicDist = [float(j) for j in flds]
	origTopicDist.append(topicDist)


matchedTopics = []
matchedTopicsDiff = []


for j in range(numTopics):

	betaTd = betaTranspose[j]

	minDiff = vocabsize

	candMatchedTopic = -1
	candMatchedTopicDiff = vocabsize

	for k in range(numTopics):

		if k not in matchedTopics:
			origtd = origTopicDist[k][1:]

			diff = [abs(origtd[i] - betaTd[i]) for i in range(vocabsize)]

			diffSum = sum(diff)

			if diffSum < minDiff:
				candMatchedTopic = k
				candMatchedTopicDiff = diffSum
				minDiff = diffSum

				# print (j, k, minDiff)

		else:
			continue

	matchedTopics.append(candMatchedTopic)
	matchedTopicsDiff.append(candMatchedTopicDiff)


print(matchedTopics)
print(matchedTopicsDiff)
