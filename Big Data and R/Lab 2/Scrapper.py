import requests
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
import  pandas as pd


# Execute one after other subtopic
def scrap_data(path):
    subtopics = ["Apple","facebook","google","microsoft", "amazon"]

    for subtopic in subtopics:
        path = path + "\\" +subtopic+"\\url.txt"
        links = []
        with open(path, "r") as f:
            links = f.readlines()
        i = 0
        print("Total links for " + subtopic + " is: " + str(len(links)))
        for link in links:
            i+=1
            content = ""
            page = requests.get(link)
            soup = BeautifulSoup(page.content, 'html.parser')
            header_html = soup.select('article header h1')
            header_text = ''
            if len(header_html) > 0:
                header_text = header_html[0].get_text()
            paras = soup.select('article p')
            content = content + header_text
            for p in range(0,len(paras)):
                if len(paras[p].get_text()) < 50:
                    continue
                content = content + paras[p].get_text()
            with open("rawdata" + "\\" +subtopic + "\\" +subtopic + "_" + str(i) + ".txt", "w", encoding="utf-8") as f:
                f.write(content)




# scrap_data()




