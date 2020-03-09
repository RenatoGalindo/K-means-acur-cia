# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 03:49:41 2020

@author: Renato
"""


import json
# Data Structures
import numpy  as np
import pandas as pd

import json

# Corpus Processing
import re
import nltk.corpus
import nltk
from unidecode                        import unidecode
nltk.download('stopwords')
import nltk
nltk.download('punkt')

from nltk.tokenize                    import word_tokenize
from nltk                             import SnowballStemmer
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.feature_extraction       import DictVectorizer
from sklearn.preprocessing            import normalize

# K-Means
from sklearn import cluster
from sklearn.cluster import KMeans

# Visualization and Analysis
import matplotlib.pyplot  as plt
import matplotlib.cm      as cm
import seaborn            as sns
from sklearn.metrics                  import silhouette_samples, silhouette_score



#importando dados

uri2  = "https://raw.githubusercontent.com/RenatoGalindo/K-means-acur-cia/master/Pergunta2.csv"

dataNew = pd.read_csv(uri2 ,error_bad_lines=False ,encoding = "ISO-8859-1")
dataNew.columns = map(str.lower, dataNew.columns)
#printando pegando as 5 primeiras linhas
print(dataNew.head(5))

#pegando a clouna pergunta
corpus = dataNew['pergunta'].tolist()
#imprindo as colunas
print(corpus)

# removes a list of words (ie. stopwords) from a tokenized list.
def removeWords(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]

# applies stemming to a list of tokenized words
def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]

# removes any words composed of less than 2 or more than 21 letters
def twoLetters(listOfTokens):
    twoLetterWord = []
    for token in listOfTokens:
        if len(token) <= 2 or len(token) >= 21:
            twoLetterWord.append(token)
    return twoLetterWord

def processCorpus(corpus, language):   
    stopwords = nltk.corpus.stopwords.words(language)
    param_stemmer = SnowballStemmer(language)


    
    for document in corpus:

        index = corpus.index(document)
        
        corpus[index] = corpus[index].replace(u'\ufffd', '8')   # Replaces the ASCII '�' symbol with '8'
        corpus[index] = corpus[index].replace(',', '')          # Removes commas
        corpus[index] = corpus[index].rstrip('\n')              # Removes line breaks
        corpus[index] = corpus[index].casefold()                # Makes all letters lowercase
        
        corpus[index] = re.sub('\W_',' ', corpus[index])        # removes specials characters and leaves only words
        corpus[index] = re.sub("\S*\d\S*"," ", corpus[index])   # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
        corpus[index] = re.sub("\S*@\S*\s?"," ", corpus[index]) # removes emails and mentions (words with @)
        corpus[index] = re.sub(r'http\S+', '', corpus[index])   # removes URLs with http
        corpus[index] = re.sub(r'www\S+', '', corpus[index])    # removes URLs with www
       
       
        listOfTokens = word_tokenize(corpus[index])
      
 
        twoLetterWord = twoLetters(listOfTokens)
        listOfTokens = removeWords(listOfTokens, stopwords)
      
       
        
        
        listOfTokens = removeWords(listOfTokens, twoLetterWord)

        
        
        corpus[index]   = " ".join(listOfTokens)
     
        
        corpus[index] = unidecode(corpus[index])
       

        

    return corpus

language = 'portuguese'

corpus = processCorpus(corpus, language)

print(corpus)

#vectorizer = TfidfVectorizer(sublinear_tf=True)
vectorizer = TfidfVectorizer()



X = vectorizer.fit_transform(corpus)

idf_values = dict(zip(vectorizer.get_feature_names(),X.toarray() ))

tf_idf = pd.DataFrame(data = X.toarray(), columns=vectorizer.get_feature_names())

final_df = tf_idf

lista_palavra = {}

listaScores = list()
listaKeys = list()
listaPalavras = list()

for key , row in final_df.iterrows():
     
    #listaScore.linha = row
    for key2 , valueloc in final_df.loc[key].items(): 
        
           
        if valueloc > 0.0 :
            
            listaKeys.append(key)
            listaPalavras.append(key2)
            listaScores.append(valueloc)
            
            lista_palavra['linha'] = listaKeys
            lista_palavra['palavra_chave'] = listaPalavras
            lista_palavra['score'] = listaScores
            
mai = []    
ultimalinha = 0       
datacsv = pd.DataFrame(lista_palavra, columns=['linha','palavra_chave','score']) 
datacsv.to_csv('C:/Users/Lilia/Documents/palavra.csv', index=False) 
contador = 0


for  linhakey,   values in datacsv.iterrows():
   # print(linhakey)
 
  
  if values['linha'] == ultimalinha:
        print(ultimalinha , mai)
        print(contador)
       
        ultimalinha = values['linha']    
        if values['score'] == 0.0 and  values['linha'] == ultimalinha: 
            mai.append(values['score'])
            print(mai[0])
            print('teste')
           
            ultimalinha = values['linha'] 
        else:
            if values['score'] > 0.0:
              mai.append(values['score'])
              contador = contador + 1
  else:
      ultimalinha = values['linha']


         
   


#teste_loc = {}
#Steste_loc = list()
#for value in final_df.items():
#   teste_loc =value 
    
  #  for key , valueloc in teste_loc.items(): 
      
     # print (valueloc)
    
       
           



    

final_df.T.nlargest(20, 0)



_clusters = 100
true_k = _clusters
model = KMeans(
        n_clusters=true_k,
        init="k-means++",
        max_iter=100,
        n_init=1
    )
model.fit(X)



dataNewCsv = {
'Pergunta':dataNew['pergunta'],


}

clusters_list = list()
for key , value in dataNew['pergunta'].items():
        
        clusters_list.append( int(model.predict(vectorizer.transform([value]))   )                         )
    
         
dataNewCsv["Cluster"]=clusters_list
datcsv = pd.DataFrame(dataNewCsv, columns=['Cluster','Pergunta','Palavra chave'])

    
datcsv.to_csv('C:/Users/Lilia/Documents/testdf.csv', index=False)