import nltk
from nltk.tokenize.regexp import WhitespaceTokenizer
from nltk import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


def getUniqueWords(allWords):
    uniqueWords = []
    for i in allWords:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords


text_str = open('corpus.txt').read()
tokens = WhitespaceTokenizer().tokenize(text_str)
print("\nInitial Statistics of the Corpus.")
print("#token: "+str(len(tokens)))
print("#types: "+str(len(getUniqueWords(tokens))))

print("\nThe Top-10 Frequent Tokens.")
freq = nltk.FreqDist(tokens)
print(freq.most_common(10))

tokens = [token.lower() for token in tokens]
print("\nAfter Case Folding.")
print("#token: "+str(len(tokens)))
print("#types: "+str(len(getUniqueWords(tokens))))

print("\nThe Top-10 Frequent Tokens.")
freq = nltk.FreqDist(tokens)
print(freq.most_common(10))

nltk.download("stopwords")
stop_words = set(stopwords.words('english'))


filtered_tokens = [w for w in tokens if not w in stop_words]

filtered_tokens = []

for w in tokens:
    if w not in stop_words:
        filtered_tokens.append(w)

print("\nRemoving Stop Words.")
print("#token: "+str(len(filtered_tokens)))
print("#types: "+str(len(getUniqueWords(filtered_tokens))))

print("\nThe Top-10 Frequent Tokens.")
freq = nltk.FreqDist(filtered_tokens)
print(freq.most_common(10))

nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
lem_tok = []
for tok in filtered_tokens:
    t = lemmatizer.lemmatize(tok)
    lem_tok.append(t)

print("\nLemmatization.")
print("#token: "+str(len(lem_tok)))
print("#types: "+str(len(getUniqueWords(lem_tok))))

print("\nThe Top-10 Frequent Tokens.")
freq = nltk.FreqDist(lem_tok)
print(freq.most_common(10))

ps = PorterStemmer()
stem_tok = []
for tokk in lem_tok:
    tt = ps.stem(tokk)
    stem_tok.append(tt)

print("\nStemming.")
print("#token: "+str(len(stem_tok)))
print("#types: "+str(len(getUniqueWords(stem_tok))))

print("\nThe Top-10 Frequent Tokens.")
freq = nltk.FreqDist(stem_tok)
lst =FreqDist(dict(freq.most_common(10)))
lst.plot()