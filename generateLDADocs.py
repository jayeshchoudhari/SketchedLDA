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
wordDistFile = open("topicWordDistribution.txt", "w")
for x in range(numTopics):
    np.random.seed()
    localWordDist = list(np.random.dirichlet(hyperAlpha))
    topicWordDist.append(localWordDist)

    writeStr = str(x) + ' ' + ' '.join([str(i) for i in localWordDist]) 

    wordDistFile.write(writeStr + "\n")

wordDistFile.close()


# Document-Topic Dist Dirichlet Parameter
hyperBeta = [0.001]*numTopics

documentFilePtr = open("LDAdocuments.txt", "w")

# generate documents...
for j in range(numDocuments):
	# generate document-topic distribution using Dirichlet 
	docTopicDist = list(np.random.dirichlet(hyperBeta))
	
	# sample document length from a Poisson
	docLength = np.random.poisson(meanDocLength)

	writeStr = ""

	for k in range(docLength):
		# sample topic for each word from the document-topic distribution
		sampledTopic = np.random.choice(allTopicIds, 1, True, docTopicDist)[0]

		# sample word from the topic distribution of the sampled topic 
		sampledWord = np.random.choice(allWordsIds, 1, True, topicWordDist[sampledTopic])[0]

		writeStr += str(sampledWord) + " "

	writeStr += "\n"

	documentFilePtr.write(writeStr)


documentFilePtr.close()

