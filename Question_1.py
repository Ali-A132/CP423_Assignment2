from collections import defaultdict
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('stopwords')

# preprocess text through making it lowercase, having it be alphanumerical and remove stopwords + single length chars
def preprocess(text):
    text = text.lower()
    text = ''.join(t for t in text if t.isalnum() or t.isspace())
    wordTokens = word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    filtered = [w for w in wordTokens if w not in stopWords]
    filtered = [w for w in filtered if len(w) > 1]

    return filtered

# create data structure for inverted index
class PositionalIndex:
    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(list))  

    def createPositionalIndex(self, folder):
        documents = []

        # iterate through files in scrappedInfo and store every word in index along with document frequency
        for filename in os.listdir(folder):
            filePath = os.path.join(folder, filename)
            try:
                with open(filePath, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                continue

            documents.append(filename)

            # preprocess text so we can iterate over words and populate index
            words = preprocess(text)
            for pos, word in enumerate(words):
                self.index[word][filename].append(pos)
        return documents

    # function to find phrase throughout data folder
    def phraseQueryProcessing(self, phrase):
        
        # preprocess and ignore if greater than 5 terms
        terms = preprocess(phrase)
        if len(terms) > 5:
            print("Phrase too long (max 5 terms).")
            return []

        # create list for all terms
        lists = []
        for term in terms:
            lists.append(self.index[term])

        # find all docs that have contained terms given from the phrase
        common = set(lists[0].keys())
        for lis in lists[1:]:
            common = common.intersection(set(lis.keys()))

        # iterate through all docs and find matching position of each term and if they follow each other
        matchingDocs = []
        for doc in common:
            positions = []
            for lis in lists:
                positions.append(set(lis[doc]))

            for pos in positions[0]:
                match = True
                # check if each subsequent term following the first
                for i in range(1, len(terms)):
                    if (pos + i) not in positions[i]:
                        match = False
                        break
                # append to docs list if there is a match
                if match:
                    matchingDocs.append(doc)
                    break

        return matchingDocs

posIndex = PositionalIndex()
posIndex.createPositionalIndex("data")

# Print index
# for word, file_positions in posIndex.index.items():
#     print(f"{word}:")
#     for file, positions in file_positions.items():
#         print(f"  {file}: {positions}")

# EXAMPLES
# cp423 is a text retreival course, black beard, I signed the paper as directed, and the lawyer took it
# results = posIndex.phraseQueryProcessing("I signed the paper as directed, and the lawyer took it gone")
# print("Documents that match the phrase: ", results)
# print(len(results), "documents found")
