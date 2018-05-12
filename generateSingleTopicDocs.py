# LDA generate synthetic docs

from __future__ import division
import sys
import numpy as np
import math
from collections import defaultdict
from collections import Counter


if len(sys.argv) < 5:
	print("Missing command line arguments")
	print("argv[1] = numTopics, argv[2] = vocabsize, argv[3] = numDocuments, argv[4] = mean document length")
	sys.exit()


numTopics = int(sys.argv[1])
vocabsize = int(sys.argv[2])
numDocuments = int(sys.argv[3])
meanDocLength = int(sys.argv[4])

allTopicIds = list(range(numTopics))
allWordsIds = list(range(vocabsize))

topicWordDist = []

# Topic-Word Dist Dirichlet Parameter
hyperAlpha = [0.1]*vocabsize

# writing the topic-word distributions to a file
wordDistFile = open("topicWordDistributionSingleTopic.txt", "w")
for x in range(numTopics):
    np.random.seed()
    localWordDist = list(np.random.dirichlet(hyperAlpha))
    topicWordDist.append(localWordDist)

    writeStr = str(x) + ' ' + ' '.join([str(i) for i in localWordDist]) 

    wordDistFile.write(writeStr + "\n")

wordDistFile.close()


# Document-Topic Dist Dirichlet Parameter
hyperBeta = [1]*numTopics

topicProportionDist = list(np.random.dirichlet(hyperBeta))

# writing topic proportions to file.... vec(w)
topicPropFilePtr = open("topicProportionSingleTopic.txt", "w")
writeStr = ' '.join([str(i) for i in topicProportionDist]) + "\n"
topicPropFilePtr.write(writeStr)
topicPropFilePtr.close()


documentFilePtr = open("singleTopicDocuments.txt", "w")

# create document as bag of words format
docWordMat = [ [0 for i in range(vocabsize)] for j in range(numDocuments) ]



# generate documents...
for j in range(numDocuments):
	# generate document-topic distribution using Dirichlet 
	
	# sample document length from a Poisson
	docLength = np.random.poisson(meanDocLength)
	# sample topic for each word from the document-topic distribution
	sampledTopic = np.random.choice(allTopicIds, 1, True, topicProportionDist)[0]

	writeStr = ""

	for k in range(docLength):

		# sample word from the topic distribution of the sampled topic 
		sampledWord = np.random.choice(allWordsIds, 1, True, topicWordDist[sampledTopic])[0]

		docWordMat[j][sampledWord] += 1

		writeStr += str(sampledWord) + " "

	writeStr += "\n"

	documentFilePtr.write(writeStr)


documentFilePtr.close()

# print docWordMat

# write matrix (bag of words) format document to file..

documentFilePtr = open("singleTopicDocuments_matrixFormat.txt", "w")

for i in range(len(docWordMat)):

	writeStr = ' '.join([str(docWordMat[i][j]) for j in range(len(docWordMat[i]))]) + "\n"

	documentFilePtr.write(writeStr)

documentFilePtr.close()



 