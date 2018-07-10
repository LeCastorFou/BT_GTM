import pandas as pd
import numpy as np

path_g = "/home/valentin/Documents/Bernis"
path_data = path_g + "/Donnees/"
path_prog = path_g + "/programmes/"

Presta = pd.read_pickle(path_data + 'Presta_Asso.pkl')
Presta
len(Presta)
len(np.unique(Presta.ActGTM))
