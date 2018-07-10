
# coding: utf-8

# In[259]:

import pandas as pd
import numpy as np
from io import StringIO


# In[156]:

df_GTM = pd.read_csv('GTM - Liste TMV Seed.csv',delimiter ='\t',encoding ='UTF-16LE')
df_seed = pd.read_csv('Vehicules - Historique Seed.csv',delimiter ='\t',encoding ='UTF-16LE')
df_seed


# In[158]:

# save en pickle
df_GTM.to_pickle('df_GTM.pkl')
df_seed.to_pickle('df_seed.pkl')


# In[369]:

# ouverture des ficher au format pickle
df_GTM = pd.read_pickle('df_GTM.pkl')
df_seed = pd.read_pickle('df_seed.pkl')


# In[166]:

# nombre d'action d'entretien théorique à mener
len(pd.unique(df_GTM['Désignation Action']))


# In[274]:

len(pd.unique(df_seed['Désignation']))


# In[377]:

# je supprime des lignes vides de la serie d'actions
df_GTM = df_GTM.dropna(subset=['Désignation Action'])


# ### CLEANING DATA

#                                        FICHIER GTM

# In[371]:

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.metrics import *


# In[388]:

stopwordsFrench = list(map(str.upper,set(stopwords.words('french')))) + list(punctuation)
stopwordsFrench.append('A')
stopwordsFrench.append('m')
stopwordsFrench.append('K')
st = LancasterStemmer()


# In[447]:

lst_action = []
for action in set(df_GTM['Désignation Action']):
    if pd.isnull(action) != True:
        action = action.replace('.', ' ')
        action = action.replace('/', ' ')
        action = action.replace('+', ' ')
        word_action = word_tokenize(action)
        stem_word_action =[words.replace(words,(st.stem(words)).upper()) for words in word_action]
        lst_action.append((' ').join(stem_word_action))


# In[543]:

#attention j'ai perdu 1 similarité !!
len(set(lst_action))
lst_action


#                                       FICHIER SEED

# In[393]:

# je supprime des lignes vides de la serie d'actions de seed si il y en a
df_seed = df_seed.dropna(subset=['Désignation'])


# In[473]:

#faire une fonction qui fait les replace + df transforme en liste
lst_action_seed = []
for action in set(df_seed['Désignation']):
    if pd.isnull(action) != True:
        action = action.replace('.', ' ')
        action = action.replace('/', ' ')
        action = action.replace('+', ' ')
        word_action = word_tokenize(action)
        stem_word_action =[words.replace(words,(st.stem(words)).upper()) for words in word_action]
        # je filtre les rows qui conduise à du vide
        if (' ').join(stem_word_action) == '':
            df_tempo = df_seed[df_seed['Désignation'] != action]
        else :
            lst_action_seed.append((' ').join(stem_word_action))


# In[ ]:




# In[628]:

# dictionnaire frequence des actions : actions
from collections import Counter
dico = Counter(lst_action_seed)
sorted_d = sorted ((value,key) for (key,value) in dico.items())
sorted_d[-30:]

# extraction de la liste des 30 actions majoritaires
action_majo = sorted_d[-30:]
lst_action_majo = [elmt[1] for elmt in action_majo]
lst_action_majo = [(elmt[1].replace(' ','')) for elmt in action_majo]
lst_action_majo


# SIMILARITE : http://eric.univ-lyon2.fr/~ricco/cours/slides/TM.B%20-%20matrice%20documents%20termes.pdf

#                                TEST AVEC EDIT DISTANCE

# In[620]:

# Calcul de la similarité avec edit distance (Levenshtein)
for word in lst_action_majo :
    for action in lst_action:
        distance_norm = edit_distance(word, action.replace(' ',''))/max([len(word),len(action.replace(' ',''))])
        if distance_norm < 0.4:
            print(word)
            print(action.replace(' ',''))
            print(distance_norm)


#                                     TEST AVEC JACCARD

# In[623]:

action_majo = sorted_d[-100:]
lst_mot_majo = [elmt[1] for elmt in action_majo]
print(len(lst_mot_majo + lst_action))
print(len(set(lst_mot_majo + lst_action)))


# In[629]:

# Construction de la matrice document-terme
from sklearn.feature_extraction.text import CountVectorizer
#instantiation of the objet – binary weighting
parseur = CountVectorizer(binary=True)
#create the document term matrix
X = parseur.fit_transform(set(lst_mot_majo + lst_action))
tokens = parseur.get_feature_names()
#print((tokens))
df = pd.DataFrame(data=X.toarray(), index=set(lst_mot_majo + lst_action),
             columns=tokens)


# In[631]:

# Extraction d'un dictionnaire final pour l'analyse termes : score
dict_final_jac={}
from sklearn.metrics import jaccard_similarity_score
for action in lst_mot_majo:
    for entretien_theo in lst_action:
        jac_score = 1 - jaccard_similarity_score(df.loc[action].values, df.loc[entretien_theo].values, normalize=True)
        if jac_score > 0.4 :
            dict_final_jac.update({(action,entretien_theo):jac_score})
dict_final_jac
