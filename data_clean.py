# clean data
import re
import csv 
import sys
from HTMLParser import HTMLParser
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


# remove all words that are meaningless
def remove_stopword(words):
    # meaningless word set
    stopWords = set(stopwords.words('english'))
    stopWords.update(("and","the","a","an","in","on","at","they","if","is","are","was","were","or","th"
                      "dir","of","documents","what","where","when","how","but","no","figure","tm","pm","th","st","say","said"
                      "could","may","need","too","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t"
                      "u","v","w","x","y","z"))

    wordsFiltered = []
    # filter out meaningless words
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    cleaned_content = ' '.join(wordsFiltered)
    return cleaned_content

# remove all html elements 
class MyHTMLParser(HTMLParser):  
    def __init__(self):
        HTMLParser.__init__(self)
        self.html_plain_article = []
    def handle_data(self, data): 
        self.html_plain_article.append(data)

def html_parser(content):
    html_parser = MyHTMLParser()
    html_parser.reset()
    # Feed the article to get rid of html elements
    html_parser.feed(content) 
    # Get the list of words
    plain_article = html_parser.html_plain_article  
    plain_article = ' '.join(plain_article)
    html_parser.html_plain_article = [] 
    return plain_article


def main():
    # open the dataset
    input_file = open("hoodline_challenge.csv", 'rb') 
    article_reader = csv.DictReader(input_file)
    output_file = open("clean_article.txt", 'w') 

    for row in article_reader:
        # remove html element
        clean_html_content = ""
        clean_html_content = html_parser(row['content'])
        # remove non-alpha component and change to lowercase
        content = re.sub("[^a-zA-Z]+", " ", clean_html_content).lower()
        # tokenize each article and remove meaningless words
        words = word_tokenize(content)
        clean_content = remove_stopword(words)
        # store the cleaned content
        output_file.write(str(row['id']) + '\t' + clean_content +'\n')

    input_file.close()
    output_file.close()


if __name__ == "__main__":
    main()

