# *********************************
# implement K means with best K 
# Input:
#        clean_article.txt
# ********************************

import pandas as pd
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# open cleaned articles, * format {id,content} *
input_file = "clean_article.txt"
corpus = open(input_file, 'r').read().split('\n')
corpus_text = []

# get content corpus from cleaned articles
for line in corpus:
    if len(line) > 0:
        l = line.split('\t')
        corpus_text.append(l[1])

# calculate tf-idf matrix
tfidf_vectorizer = TfidfVectorizer(tokenizer= word_tokenize, max_features=2000, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_text)
keywords = tfidf_vectorizer.get_feature_names()

# implement K means with best K values
best_K= 3
KMmodel = KMeans(best_K)  
KMmodel.fit(tfidf_matrix)  
clusters = KMmodel.labels_.tolist()

# calulate number of document in each cluster
articles = {'text': corpus_text,'cluster': clusters} 
frame = pd.DataFrame(articles, index = [clusters], columns = ['cluster'])  
print('Number of documents per cluster')
frame['cluster'].value_counts()  

# store all words in data frame
words = [] 
for i in corpus_text:
    word = word_tokenize(i)   
    words.extend(word)  
vocab_frame = pd.DataFrame({'words': words}, index = words)

# get ordered centroid
order_centroids = KMmodel.cluster_centers_.argsort()[:, ::-1] 

# get top 10 key words according to each cluser
for i in range(best_K):
    print(("Cluster %s top words:") % i)
    for indice in order_centroids[i, :10]:
        word = vocab_frame.ix[keywords[indice].split(' ')].values.tolist()[0][0]
        print('%s' % word)
