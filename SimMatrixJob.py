import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import nltk
from nltk.corpus import stopwords
import re
#from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
#from string import punctuation
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.metrics import *


path_g = "/home/valentin/Documents/Bernis"
path_data = path_g + "/Donnees/"
path_prog = path_g + "/programmes/"

exec(open(path_prog+'Helper.py').read())

GTM = pd.read_csv(path_data + "GTM-ListeMVSeed_T.csv",sep=';',decimal=b',')
Veh = pd.read_csv(path_data + "Vehicules-HistoriqueSeed_T.csv",sep=';',decimal=b',')
#df_seed = pd.read_csv('Vehicules - Historique Seed.csv',delimiter ='\t',encoding ='UTF-16LE')




Veh['VIN'] = Veh['VIN'].astype(str)
GTM['VIN'] = GTM['VIN'].astype(str)


Veh.columns = np.array(['CodeClient', 'Compteur2', 'Compteur', 'Constructeur', 'Date','Designation', 'Entreprise', 'Etablissement', 'Immatriculation', 'LibelleCommercial', 'LibelleModele', 'Marque', 'MontantHT','Nature', 'NoLigne', 'NoOR', 'Nombredenregistrements', 'PGI','PrixUnitaire', 'Quantite', 'Reference', 'Temps', 'Type', 'VIN'])
Veh['Designation'] = Veh['Designation'].astype(str)
Veh['Date'] = pd.to_datetime(Veh['Date'], format="%d/%m/%Y")

GTM.columns = np.array(['Action', 'AnneePrevisionnelle', 'ClientFacturation', 'Client','CodeAction', 'CodeEntrepriseEntretien',
'CodeEntrepriseRealisation', 'CodeEntrepriseVehicule',
'CodeEtablissementEntretien', 'CodeEtablissementRealisation',
'CodeEtablissementVehicule', 'DateCreationTMV', 'DateDerniereACM',
'DateMEC', 'DatePrevisionnelle', 'DesignationAction',
'DesignationVehicule', 'EtablissementEntretien',
'EtablissementRealisation', 'EtablissementVehicule', 'Frequence',
'HeureFutureACM', 'Heures', 'Immatriculation', 'KMDerniereACM',
'KMFutureACM', 'MoisPrevisionnel', 'Montant', 'Moyenne',
'NomClientFacturation', 'NomClient', 'Nombredenregistrements',
'NClientFacturation', 'NClient', 'OaCamo', 'OaHf',
'SiteRealisation', 'Statut', 'TypeAction', 'VIN'])

# On ne garde que les lignes pour lesquelles VIN commence par VF
Veh = Veh[ [l.startswith("VF") for l in Veh["VIN"]] ]
# On enleve les valeurs null
Veh = Veh[Veh['Designation'].notnull()]
Veh = Veh[Veh['Designation']!='nan']
# On enleve les avoirs
Veh = Veh[Veh['MontantHT']>=0]

# On enleve les Designations rempli de characteres speciaux
Veh = Veh[ [bool(re.search('[a-zA-Z]',l)) for l in Veh["Designation"]] ]


Veh['Designation'] = Veh['Designation'].astype(str)
Veh['Designation'] = Veh['Designation'].str.upper()

################## TEST ###################

#len(Veh)
#action_unique = pd.unique(Veh['Designation'])

#stopwordsFrench = list(map(str.upper,set(stopwords.words('french')))) + list(punctuation)
#stopwordsFrench.append('A')
#stopwordsFrench.append('m')
#stopwordsFrench.append('K')
#st = LancasterStemmer()

#lst_action_seed = []
#for action in action_unique :
#    if pd.isnull(action) != True:
#        action = action.replace('.', ' ')
#        action = action.replace('/', ' ')
#        action = action.replace('+', ' ')
#        word_action = word_tokenize(action)
#        stem_word_action =[words.replace(words,(st.stem(words)).upper()) for words in word_action]
        # je filtre les rows qui conduise à du vide
#        if (' ').join(stem_word_action) == '':
#            df_tempo = Veh[Veh['Designation'] != action]
#        else :
#            lst_action_seed.append((' ').join(stem_word_action))

#len(df_tempo)
#len(lst_action_seed)
#len(action_unique)
#action_unique = pd.DataFrame(action_unique)
#action_unique['corres'] = lst_action_seed

######### Liste des actions de maintenance #########
ActGTM = list(GTM['DesignationAction'])
ActGTM = [ str(w) for w in ActGTM]
ActGTM = list(set(ActGTM))
ActGTM = np.array(ActGTM)
# On selectionne une presta en particulier ######
#ActGTM = ActGTM[14]
ActGTM_str = re.sub(' ', '',str(ActGTM))
ActGTM = ActGTM[ActGTM != "nan"]
#ActGTM_Red = np.array(['CONTROLE TECHNIQUE','CONTROLE CLIM',' REMPL POMPE A EAU','VIDANGE MOTEUR'])

Des = list(Veh['Designation'])
Des = [str(w) for w in Des]



#### Matrice pour stocker les résultats
S = (np.size(ActGTM),len(Des))
Sim_Matrix = np.zeros(S)


for i in range(0,len(ActGTM)):
    print(i,'   ',ActGTM[i])
    Sim_Matrix[i] = [text_cosine(ActGTM[i],w) for w in Des]

for i in range(0,len(ActGTM)):
    Veh[ActGTM[i]] = Sim_Matrix[i]


Veh.to_pickle(path_data + 'VehSimMatrix.pkl')
