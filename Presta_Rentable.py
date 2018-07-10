import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re

plotly.tools.set_credentials_file(username='valentin.lefranc', api_key='Xb0HU4LGnX8h3COhUgJr')

path_g = "/home/valentin/Documents/Bernis"
### Sur le serveur
#path_g = "/home/seed/Projet_POC_GTM"

path_data = path_g + "/Donnees/"
path_prog = path_g + "/programmes/"

exec(open(path_prog+'Helper.py').read())

Veh = pd.read_pickle(path_data + 'VehSimMatrix.pkl')

ActGTM = Veh.columns[24:]
ActGTM = [ str(w) for w in ActGTM]
ActGTM = list(set(ActGTM))

Presta_rentable = []

for act in ActGTM:
    Veh_sub = Veh[Veh[act] >0.5]
    if np.sum(Veh_sub['Temps'])>0 :
        Presta_rentable.append(np.sum(Veh_sub['MontantHT'])/np.sum(Veh_sub['Temps']))
    else:
        Presta_rentable.append(0)
        #print( act + '   ' + str(np.sum(Veh_sub['MontantHT'])/len(Veh_sub) ) )
Presta = pd.DataFrame(ActGTM)
Presta_rentable = [x/max(Presta_rentable) for x in Presta_rentable]
Presta['rent'] = Presta_rentable
len(Presta_rentable)


Presta = Presta.sort_values(by =['rent'], ascending=False)
Presta.to_csv(path_data + 'Classement_Presta_Rentable.csv')
