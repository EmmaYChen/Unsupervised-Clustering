# ***********************************************************
# implement NMF to find meaningfyl topics in a corpus
# Input:
#        clean_article.txt
# **********************************************************

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

#function displays top words for a Topic
def display_topics(model, feature_names, number_top_words):
    for topic_index, topic in enumerate(model.components_):
        print ("Keywords of topic %d:" % (topic_index))
        print (",".join([feature_names[i] for i in topic.argsort()[:-number_top_words - 1:-1]]))

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
    keywords = tfidf_vectorizer.get_feature_names()
    
    #three is selected as there are three clusters
    number_topics = 3  

    # Run Non-negative Matrix Factorization (NMF) 
    nmf = NMF(n_components=number_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf_matrix)
    number_top_words = 10
    display_topics(nmf, keywords, number_top_words)

if __name__ == "__main__":
    main()





