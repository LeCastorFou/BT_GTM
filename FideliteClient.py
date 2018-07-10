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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import preprocessing

plotly.tools.set_credentials_file(username='valentin.lefranc', api_key='Xb0HU4LGnX8h3COhUgJr')

#path_g = "C:/Users/Lefranc/Documents/Dossier_Client_Temporaire/BernisInvestissement2017-6BT-GTM/"
#path_data = path_g + "donnees/"
#path_prog = path_g + "programmes/"

#path_os = "/Volumes/Commun/"
#path_win = "Q:/"

#path_g = "ProjetsClients/BernisInvestissement/BernisInvestissement2017-6BT-GTM/"
#path_g = path_os + "ProjetsClients/BernisInvestissement/BernisInvestissement2017-6BT-GTM/"
#path_data = path_g + "donnees/"
#path_prog = path_g + "programmes/"

# En local
path_g = "/home/valentin/Documents/Bernis"
path_data = path_g + "/Donnees/"
path_prog = path_g + "/programmes/"


exec(open(path_prog+'Helper.py').read())

GTM = pd.read_csv(path_data + "GTM-ListeMVSeed_T.csv",sep=';',decimal=b',')
#Veh = pd.read_csv(path_data + "Vehicules-HistoriqueSeed_T.csv",sep=';',decimal=b',')
Veh = pd.read_pickle(path_data + 'VehSimMatrix.pkl')

Veh.head()
Veh.columns


Veh['VIN'] = Veh['VIN'].astype(str)
GTM['VIN'] = GTM['VIN'].astype(str)

#Veh.columns = np.array(['CodeClient', 'Compteur2', 'Compteur', 'Constructeur', 'Date','Designation', 'Entreprise', 'Etablissement', 'Immatriculation', 'LibelleCommercial', 'LibelleModele', 'Marque', 'MontantHT','Nature', 'NoLigne', 'NoOR', 'Nombredenregistrements', 'PGI','PrixUnitaire', 'Quantite', 'Reference', 'Temps', 'Type', 'VIN'])
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



##########################################
########## Data des actions GTM ##########
##########################################

ActGTM = Veh.columns[24:len(Veh.columns)]

#a = Veh[Veh['VIN' ] == 'VF622ACB000101129'  ]
#a = a[a['NoOR'] == 1061915]


ActGTM

## Verifier ce qu'il se passe (les longeurs sont differente mais les vecteurs réduits semblables)
I_ActGTM = Veh[ActGTM].max(1)

#Veh.head()
#len(I_ActGTM)
#len(Veh)
Veh['Is_ActGTM'] = I_ActGTM
Veh['Is_ActGTM'][Veh['Is_ActGTM'] > 0.7] = 1
Veh['Is_ActGTM'][Veh['Is_ActGTM'] < 0.7] =  0


Veh_GTM = Veh[list(I_ActGTM >0.7) ]
#Veh_GTM.to_pickle(path_data + 'Veh_GTM.pkl')
Veh_GTM = pd.read_pickle(path_data + 'Veh_GTM.pkl')
len(Veh_GTM)/6100000
#len(Veh_GTM)
#np.unique(Veh_GTM['Designation'])[1:30]

# Nombre d'ACM GTM
Top_Veh_GTM = nocu(Veh_GTM,'VIN')
Top_Veh_GTM['VIN'] = Top_Veh_GTM.index
Top_Veh_GTM = Top_Veh_GTM.sort_values(by =['VIN'])

### Sum MontantHT
Evol_HT = sumcu(Veh_GTM,'VIN','MontantHT')
Evol_HT['VIN'] = Evol_HT.index
Evol_HT = Evol_HT.sort_values(by =['VIN'])

# Nb d'heure
Evol_Heure = sumcu(Veh_GTM,'VIN','Quantite')
Evol_Heure['VIN'] = Evol_Heure.index
Evol_Heure = Evol_Heure.sort_values(by =['VIN'])

## Nb KM
Max_KM = maxcu(Veh_GTM,'VIN','Compteur')
Max_KM['VIN'] = Max_KM.index
Max_KM = Max_KM.sort_values(by =['VIN'])

# Age du Veh en jour
Age_Veh = DiffDate(Veh_GTM,'VIN','Date')
Age_Veh['VIN'] = Age_Veh.index
Age_Veh = Age_Veh[Age_Veh['VIN'].isin(Veh_GTM['VIN'])]
Age_Veh = Age_Veh.sort_values(by =['VIN'])

## Data frame intermediaire avec les info sur les ACM GTM
Age_Veh = Age_Veh[['VIN','AgeInDays']]
Age_Veh['maxkm'] = Max_KM['max']
Age_Veh['sumHT'] = Evol_HT['sum']
Age_Veh['no'] = Top_Veh_GTM['no']
Age_Veh['Heure'] = Evol_Heure['sum']

Age_Veh['PrixAuJour'] = Age_Veh['sumHT']/Age_Veh['AgeInDays']
Age_Veh['HparJour'] = Age_Veh['Heure']/Age_Veh['AgeInDays']
Age_Veh['Freq'] = Age_Veh['no']/Age_Veh['AgeInDays']

Age_Veh = Age_Veh[Age_Veh['AgeInDays'] > 0]

###############################################
##### Les ACM non inscrites dans GTM  #########
###############################################
Veh_NO_GTM = Veh[list(I_ActGTM < 0.7) ]
#Veh_NO_GTM.to_pickle(path_data + 'Veh_NO_GTM.pkl')
# Ou pour charger un version pre compile
Veh_NO_GTM = pd.read_pickle(path_data + 'Veh_NO_GTM.pkl')
Veh_NO_GTM = Veh_NO_GTM[Veh_NO_GTM['VIN'].isin(Age_Veh['VIN'])]
len(Veh_NO_GTM)

# Nombre d'ACM
Top_Veh_NO_GTM = nocu(Veh_NO_GTM,'VIN')
Top_Veh_NO_GTM['VIN'] = Top_Veh_NO_GTM.index
Top_Veh_NO_GTM = Top_Veh_NO_GTM.sort_values(by =['VIN'])
Top_Veh_NO_GTM.columns = np.array(['no_NO','index','VIN'])
Top_Veh_NO_GTM = Top_Veh_NO_GTM[['VIN','no_NO']]


#### sum du MontantHT
Evol_HT_NO = sumcu(Veh_NO_GTM,'VIN','MontantHT')
Evol_HT_NO['VIN'] = Evol_HT_NO.index
Evol_HT_NO = Evol_HT_NO.sort_values(by =['VIN'])
Evol_HT_NO.columns = np.array(['sumHT_NO','index','VIN'])
Evol_HT_NO = Evol_HT_NO[['VIN','sumHT_NO']]

#### Sum des Heures
Evol_Heure_NO = sumcu(Veh_NO_GTM,'VIN','Quantite')
Evol_Heure_NO['VIN'] = Evol_Heure_NO.index
Evol_Heure_NO = Evol_Heure_NO.sort_values(by =['VIN'])
Evol_Heure_NO.columns = np.array(['Heure_NO','index','VIN'])
Evol_Heure_NO = Evol_Heure_NO[['VIN','Heure_NO']]
######################################################################

temp = pd.merge(Age_Veh, Evol_Heure_NO, on='VIN')
temp = pd.merge(temp, Evol_HT_NO, on='VIN')
temp = pd.merge(temp, Top_Veh_NO_GTM, on='VIN')

temp['PrixAuJour_NO'] = temp['sumHT_NO']/temp['AgeInDays']
temp['HparJour_NO'] = temp['Heure_NO']/temp['AgeInDays']
temp['Freq_NO'] = temp['no_NO']/temp['AgeInDays']
temp

Age_Veh_classif = temp
Age_Veh_classif.columns

# Remove outliers
Age_Veh_classif = Age_Veh_classif[ list(Age_Veh_classif['Freq'] > np.percentile(Age_Veh_classif['Freq'],10) ) and list(Age_Veh_classif['Freq'] < np.percentile(Age_Veh_classif['Freq'],90)) ]
Age_Veh_classif = Age_Veh_classif[ list(Age_Veh_classif['sumHT_NO'] < np.percentile(Age_Veh_classif['sumHT_NO'],99)) ]

Age_Veh_classif_cr = Age_Veh_classif[['sumHT','HparJour','PrixAuJour','Freq','sumHT_NO','HparJour_NO','PrixAuJour_NO','Freq_NO','AgeInDays']]


x = Age_Veh_classif_cr.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
Age_Veh_classif_cr = pd.DataFrame(x_scaled)


Age_Veh_classif_cr.columns = ['sumHT','HparJour','PrixAuJour','Freq','sumHT_NO','HparJour_NO','PrixAuJour_NO','Freq_NO','AgeInDays']
#Age_Veh_classif_cr

#### On passe certaine variable en longeurs
#Age_Veh_classif_cr['HparJour'] = np.log(Age_Veh_classif_cr['HparJour']+1)
#Age_Veh_classif_cr['HparJour_NO'] = np.log(Age_Veh_classif_cr['HparJour_NO']+1)
#Age_Veh_classif_cr['PrixAuJour'] = np.log(Age_Veh_classif_cr['PrixAuJour']+1)
#Age_Veh_classif_cr['PrixAuJour_NO'] = np.log(Age_Veh_classif_cr['PrixAuJour_NO']+1)

#generer la matrice des liens
Z = linkage(Age_Veh_classif_cr,method='ward',metric='euclidean')
#affichage du dendrogramme

dendrogram(Z,labels=Age_Veh_classif_cr.index,orientation='left',color_threshold=6)
plt.show()

#découpage à la hauteur t = 7 ==> identifiants de 4 groupes obtenus
groupes_cah = fcluster(Z,t=7,criterion='distance')
Age_Veh_classif_cr['groupes'] = groupes_cah



#######################################
####### PLOTS #########################
#######################################
Age_Veh_classif_cr.boxplot(by ='groupes', showfliers=False,figsize=(5,5))


Age_Veh_classif_cr.describe()
Age_Veh_classif['groupes'] = groupes_cah

Age_Veh_classif.columns
Age_Veh_classif['HparAn'] = Age_Veh_classif['HparJour']*250
Age_Veh_classif['groupes']

np.max(Age_Veh_classif[Age_Veh_classif['groupes'] == 1])
np.max(Age_Veh_classif['HparJour'])*250


import seaborn as sns

#sns.kdeplot(Age_Veh_classif_cr['HparJour'])

#len(np.unique(Age_Veh_classif['VIN']))
#plt.scatter(Age_Veh_classif_cr['sumHT'],Age_Veh_classif_cr['sumHT_NO'])
#plt.scatter(Age_Veh_classif_cr['Freq'],Age_Veh_classif_cr['Freq_NO'])

################
## Sauvegarde ##
################
#Age_Veh_classif_f = Age_Veh_classif_cr
#Age_Veh_classif_f['VIN'] =  Age_Veh_classif['VIN']
#Age_Veh_classif_f.to_csv(path_data + "Age_Veh_classif.csv",sep=',')


Veh_Groupe = Age_Veh_classif_cr[['VIN','groupes']]

Veh_IsGTM = pd.merge(Veh, Veh_Groupe, on='VIN')

Veh_IsGTM.to_pickle(path_data + 'Veh_ACM.pkl')

Age_Veh_classif_cr.to_pickle(path_data + 'Veh_classif_cr.pkl')
Age_Veh_classif_cr.to_csv(path_data + 'Veh_classif_cr.csv')

###########################################
## Le vehicule parfait  ###################
###########################################

len(Age_Veh_classif)
len(Age_Veh_classif_cr)
