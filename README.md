# SentimentAnalysis_with_TextBlob
A template for sentiment analysis with the library TextBlob

"""
Created on Wed Oct 14 13:03:54 2020

@author: frgentile
"""

    import csv
    import pandas as pd
    import numpy as np
    import googletrans
    from googletrans import Translator
    import nltk
    from nltk import word_tokenize
    from textblob import TextBlob


    original_path = "Valutazione_Anonimizzato_original.csv"
    original = pd.read_csv(original_path, encoding='latin-1')


    file_path = "Valutazione_Anonimizzato.csv"
    file_path_2 = "Valutazione_Anonimizzato - 101-200.csv"

    first = pd.read_csv(file_path, encoding='latin-1')
    second = pd.read_csv(file_path_2, encoding='latin-1')


#Translate text to English
    
    translator = Translator()

    valutazione_inv_1 = []
    for i in first['ANVALUE']:
    valutazione_inv_1.append(i)
    trans_text_inv_1 = [(translator.translate(str(i))).text for i in valutazione_inv_1] 


    valutazione_inv_2 = []
    for i in second['ANVALUE']:
    valutazione_inv_2.append(i)
    trans_text_inv_2 = [(translator.translate(str(i))).text for i in valutazione_inv_2] 


    trans_text_inv_partI = trans_text_inv_1 + trans_text_inv_2  


#Cleaning text data
    
    stopwords_en = nltk.corpus.stopwords.words('english')
    stopwords_en.append('always')
    stopwords_en.append('way')
    stopwords_en.append('take')
    stopwords_en.append('look')
    stopwords_en.append('year')

    trans_text_inv_partI_tokenized = [word_tokenize(i.lower()) for i in trans_text_inv_partI]  #Tokenize

    def remove_stopwords(tokenized_text):    
        text = [word for word in tokenized_text if word not in stopwords_en]
        return text

    trans_text_inv_partI_tokenized2 = [remove_stopwords(l) for l in trans_text_inv_partI_tokenized] #Removes stopwords

    trans_text_inv_partI_final = [' '.join(map(str, s)) for s in trans_text_inv_partI_tokenized]  #Returns evaluations without stopwords

#first['trans_text_inv_partI_final'] = trans_text_inv_final


#Generates sentiment scores
    
    polarity_text_inv = [round(TextBlob(str(i)).sentiment.polarity, 3) for i in trans_text_inv_partI_final] 
    subjectivity_text_inv = [round(TextBlob(str(i)).sentiment.subjectivity, 3) for i in trans_text_inv_partI_final] 


#Rescales the scores
    
    from __future__ import division

    def rescale(values, new_min = 0, new_max = 10):
        output = []
        old_min, old_max = -1, 1

        for v in values:
            new_v = round(((new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min), 2)
            output.append(new_v)

        return output


    new_polarity = rescale(polarity_text_inv, new_min = 0, new_max = 10)

    def rescale_2(values, new_min = 0, new_max = 10):
        output = []
        old_min, old_max = 0, 1

        for v in values:
            new_v = round(((new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min), 2)
            output.append(new_v)

        return output

    new_subjectivity = rescale_2(subjectivity_text_inv, new_min = 0, new_max = 10)

    original['new_polarity']=new_polarity

    original['new_subjectivity']=new_subjectivity

    original.to_csv (r'C:\Users\frgentile\Desktop\Sentiment Analysis\export_dataframe91.csv', index = False, header=True, encoding = 'latin-1')


