#GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
#cell array of the words
#   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt
#   and returns a cell array of the words in vocabList.

def getVocabList():
    ## Read the fixed vocabulary list
    fid = open('vocab.txt','r')
    # Store all dictionary words in cell array vocab{}
    #n = 1899  # Total number of words in the dictionary

    # For ease of implementation, we use a struct to map the strings => integers
    # In practice, you'll want to use some form of hashmap
    vocabDict = dict()
    vocabList = list()

    for line in fid:
        (val, word) = line.split()
        vocabDict[word] = int(val)
        vocabList.append(word)

    return vocabDict, vocabList
