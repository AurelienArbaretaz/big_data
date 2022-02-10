# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:55:33 2022

@author: clede
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import sklearn
import sklearn.feature_selection
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
import sklearn
import sklearn.feature_selection

firstdf = pd.read_csv("data.csv")

#nltk.download('wordnet')
#nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
# Lets preprocess the text


def preprocess(sentence):
    sentence=str(sentence)
    # Lowercase text
    sentence = sentence.lower()
    # Remove whitespace
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    # Remove weblinks
    rem_url=re.sub(r'http\S+', '',cleantext)
    # Remove numbers
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    # Remove StopWords

    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('french')]
    #Use lemmatization
    #lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(filtered_words)

firstdf['Resume_pre']=firstdf['Resume'].map(lambda s:preprocess(s))

# Lets remove all rows with 0 ...
firstdf = firstdf[firstdf["PrixNuitee"] != 0]
dfZeroToThreeHundred = firstdf[firstdf.PrixNuitee < 300]

dfForTF = dfZeroToThreeHundred.dropna(subset = ["Resume_pre"])

DFPrixNuitee=dfForTF["PrixNuitee"]
F = dfForTF.drop(columns=['PrixNuitee', 'prix_nuitee','Identifiant'])

cv=TfidfVectorizer() 
y = DFPrixNuitee
Col_Resume = cv.fit_transform(F['Resume_pre'].values.astype('U'))
X_names = cv.get_feature_names_out()
                            
p_value_limit = 0.9999999999
dtf_features = pd.DataFrame()
for cat in np.unique(y):
    chi2, p = sklearn.feature_selection.chi2(Col_Resume, y==cat)
    dtf_features = dtf_features.append(pd.DataFrame(
                   {"feature":X_names, "score":1-p, "y":cat}))
    dtf_features = dtf_features.sort_values(["y","score"], 
                    ascending=[True,False])
    dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
X_names = dtf_features["feature"].unique().tolist()

vectorizer = TfidfVectorizer(vocabulary=X_names)
vectors = vectorizer.fit_transform(F["Resume_pre"].values.astype('U'))
dic_vocabulary = vectorizer.vocabulary_

dense = vectors.todense()
denselist = dense.tolist()
dfTFIDF = pd.DataFrame(denselist, columns=X_names)

X = F.select_dtypes(include=np.number)
X = X[['Longitude','Latitude','television_cable','wifi','seche_cheveux','monoxyde_carbone_detect','salle_sport','fer_repasser','parking_sur-place','extincteur','Climatisation','Jacuzzi','machine_laver','Capacite_accueil', 'NbChambres', 'frais_menage', 'Caution', 'frais_menage','seche_linge','cheminee_interieur','pourEnfants_famille','Piscine','NbLits','television']]
dfTFIDF = dfTFIDF.select_dtypes(include=np.number)
XTF = pd.concat([X.reset_index(drop=True), dfTFIDF.reset_index(drop=True)], axis=1)

X_train, X_test, y_train, y_test = train_test_split(XTF, y,test_size = 0.2, random_state=0)

X_test.to_csv("Xtest2.csv", index=False)

LatEtLon = X_test.loc[:, ['Latitude', 'Longitude']]
X_test = X_test.drop(['Longitude', 'Latitude'], axis=1)
X_train = X_train.drop(['Longitude', 'Latitude'], axis=1)

model_bordeaux = Ridge(alpha=0.1); #SVC(kernel="linear", C=100);  
model_bordeaux.fit(X_train, y_train)

y_predicted = model_bordeaux.predict(X_test)

X_test['PrixNuitee'] = y_predicted
X_test['Latitude'] = LatEtLon['Latitude']
X_test['Longitude'] = LatEtLon['Longitude']
X_test.to_csv("Xtest2AvecPrix.csv", index=False)



