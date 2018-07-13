import pandas as pd
import numpy as np

path_g = "/home/valentin/Documents/Bernis"
path_data = path_g + "/Donnees/"
path_prog = path_g + "/programmes/"

Presta = pd.read_pickle(path_data + 'Presta_Asso.pkl')
Presta
len(Presta)
len(np.unique(Presta.ActGTM))
L = Presta['Action']
L = list(L)
L = [s.replace(',', ' ') for s in L]
L = [s.replace('\n', ' ') for s in L]
L = [s.replace('\r', ' ') for s in L]
L = [s.replace('\s+', ' ') for s in L]
Presta['Action'] = L
Presta = Presta[Presta['counts'] >2]
Presta.to_csv(path_data + 'Presta_Asso.csv')

Presta

L[1]


Veh = pd.read_pickle(path_data + 'VehSimMatrix.pkl')
