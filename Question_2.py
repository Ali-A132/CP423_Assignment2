from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as math
from Question_1 import *

class TF_IDF():
    # initialize documents in "data", and all terms with its positional index 
    def __init__(self, terms, documents):
        self.docs = documents
        self.terms = terms
        self.termKeys = terms.keys() # to iterate through term names

    def createTfIdf(self, weight):
        N = len(self.docs)
        matrix = []

        for doc in self.docs:
            row = []

            # populate with term values of each word
            tfValues = []
            for term in self.termKeys:
                if doc in self.terms[term]:
                    tfValues.append(len(self.terms[term][doc]))
                else:
                    tfValues.append(0)

            # find max value from list of term freq vals and also the total # of terms
            maxTf = max(tfValues)
            totalTerms = sum(tfValues)

            for i, term in enumerate(self.termKeys):
                rawCount = tfValues[i] # term frequencey
                if weight == 1: # Binary
                    if doc in self.terms[term]:
                        tf = 1
                    else:
                        tf = 0
                elif weight == 2: # Raw Count
                    tf = rawCount
                    # if term == "away":
                    #     print(rawCount)
                elif weight == 3: # Normalized TF
                    tf = rawCount / totalTerms
                elif weight == 4: # Log Normalization
                    tf = math.log(1 + rawCount)
                elif weight == 5: # Double Normalization
                    tf = 0.5 + 0.5 * (rawCount / maxTf)

                df = len(self.terms[term])
                idf = math.log10(N / (df + 1)) 
                row.append(tf * idf)

            matrix.append(row)
        return np.array(matrix), self.termKeys
    
    def queryProcessing(self, query, weight):
        query = preprocess(query)
        N = len(self.docs)

        tfValues = []
        for term in self.termKeys:
            count = query.count(term)
            tfValues.append(count)

        maxTf = max(tfValues)
        totalTerms = sum(tfValues)
        row = []
        for i, term in enumerate(self.termKeys):
            rawCount = tfValues[i] # term frequencey
            if weight == 1: # Binary
                if rawCount > 0:
                    tf = 1
                else:
                    tf = 0
            elif weight == 2: # Raw Count
                tf = rawCount
                # if term == "away":
                #     print(rawCount)
            elif weight == 3: # Normalized TF
                tf = rawCount / totalTerms
            elif weight == 4: # Log Normalization
                tf = math.log(1 + rawCount)
            elif weight == 5: # Log Normalization
                tf = 0.5 + 0.5 * (rawCount / maxTf)

            df = len(self.terms[term])
            idf = math.log10(N / (df + 1)) 
            row.append(tf * idf)

        return np.array(row), self.termKeys
    
def cosineSimilarityMatrix(matrix):
    # initialize empty matrix and find norm of each vector
    simMatrix = np.zeros((len(matrix), len(matrix)))
    norm = np.linalg.norm(matrix, axis = 1)

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            # if each contains a value of 0, then no similarity, otherwise we compute cosine similairy with dot product
            if norm[i] == 0 or norm[j] == 0:
                simMatrix[i][j] = 0
            else:
                simMatrix[i][j] = np.dot()
    return simMatrix

# Create index
positionalIndex = PositionalIndex()
docs = positionalIndex.createPositionalIndex("data")

# TF-IDF Matrix
pd.set_option('display.max_rows', None)

# for weight in range(1, 6):
#     print(f"\nWeight Scheme {weight}")
#     tfIdf = TF_IDF(positionalIndex.index, docs)
#     matrix, terms = tfIdf.createTfIdf(weight)
#     df = pd.DataFrame(matrix, index = docs, columns = terms)

#     # only show columns starting with "a" since there are too many columns
#     columns = []
#     for col in df.columns:
#         if col.startswith("a"):
#             columns.append(col)
#     dfFiltered = df[columns]
#     print(dfFiltered)

# Query Processing
# model = TF_IDF(positionalIndex.index, docs)
# for w in range(1, 6):
#     query, terms = model.queryProcessing("black beard", w)
#     print(f"Weighting scheme {w}:")
#     print(query)
#     print("\n")
