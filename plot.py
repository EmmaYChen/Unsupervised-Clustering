# *********************************
# visualize K means with principle component analysis (PCA)
# Input:
#        clean_article.txt
# ********************************
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.decomposition import PCA
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl

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

# retrieve K means model
best_K = 3
model = KMeans(best_K) 
model.fit(tfidf_matrix)  

# PCA : extract the two most representative features of the data 
pca = PCA(n_components=2).fit(tfidf_matrix.todense())  #convert sparse tf-idf to dense form
data2D = pca.transform(tfidf_matrix.todense())  # get 2-D representation

legendlist=[]
legendlist.append('Travel') 
legendlist.append('Safety')
legendlist.append('Recommendation')

#plot PCA
for i in range(0, data2D.shape[0]):  
    if model.labels_[i] == 0:
        c1 = plt.scatter(data2D[i,0],data2D[i,1],c='#1b9e77', marker='+')
    elif model.labels_[i] == 1:
        c2 = plt.scatter(data2D[i,0],data2D[i,1],c='#a3061e',marker='o')
    elif model.labels_[i] == 2:
        c3 = plt.scatter(data2D[i,0],data2D[i,1],c='#7570b3',marker='*')

pl.title('K-means clusters of the articles')

#locate means of each cluster
centers2D = pca.transform(model.cluster_centers_)   
plt.scatter(centers2D[:,0], centers2D[:,1], marker='x', s=200, linewidths=3, c='r')
out_png = './pca'
plt.savefig(out_png, dpi=150)