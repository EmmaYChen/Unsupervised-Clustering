# ***************************************************************
#  calculate tf-idf and search best for K means clustering
# Input:
#        clean_article.txt
# Output
#        tfidf_vectorizer.npy 
#        tfidf_matrix.npy
#        kmeans.png
# **************************************************************

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.cluster import KMeans



def top_tfidf(tfidf_vectorizer,tfidf_matrix):
    # get top 50 key word with highest tfidf 
    n = 50
    keywords = tfidf_vectorizer.get_feature_names()
    feature_list = np.array(keywords)
    sorted_tfidf = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_n = feature_list[sorted_tfidf][:50]


def main():

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
    
    # K-means clustering, test for K from 1 to 10 clusters to get best cluster numbers
    num_clusters = range(1,8)  
    KM = [KMeans(n_clusters=k).fit(tfidf_matrix) for k in num_clusters] 
    centroids = [k.cluster_centers_ for k in KM]  
    
    # calculate the sum of squared errors for each K means
    Distance = [cdist(tfidf_matrix.toarray(), c ,'euclidean') for c in centroids]  
    eucl_dist = [np.min(D,axis=1) for D in Distance]  
    avg = [sum(d)/tfidf_matrix.shape[0] for d in eucl_dist]
    
    # plot elbow to get the best cluster numbers
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(num_clusters, avg, 'b*-')

    # elbow point
    kIdx = 2 
    ax.plot(num_clusters[kIdx], avg[kIdx], marker='o', markersize=12, 
        markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within cluster sum of squares measure')
    plt.title('Elbow for KMeans clustering')
    out_png = './kmeans'
    plt.savefig(out_png, dpi=150)

    # Save tfidf_vectorizer and tfidf_matrix
    np.save("tfidf_vectorizer", tfidf_vectorizer)
    np.save("tfidf_matrix", tfidf_matrix)


if __name__ == "__main__":
    main()

