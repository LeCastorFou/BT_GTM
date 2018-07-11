#import plotly
#import plotly.plotly as py
#import plotly.graph_objs as go
#import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
#import matplotlib.pyplot as plt
#from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
#from sklearn import preprocessing
#import seaborn as sns
from random import *
import math
#plotly.tools.set_credentials_file(username='xxx', api_key='???????')

path_g = "/home/valentin/Documents/Bernis"
### Sur le serveur
#path_g = "/home/seed/Projet_POC_GTM"

path_data = path_g + "/Donnees/"
path_prog = path_g + "/programmes/"


exec(open(path_prog+'Helper.py').read())

GTM = pd.read_csv(path_data + "GTM-ListeMVSeed_T.csv",sep=';',decimal=b',')
Veh = pd.read_csv(path_data + "Vehicules-HistoriqueSeed_T.csv",sep=';',decimal=b',')


Veh.columns = np.array(['CodeClient', 'Compteur2', 'Compteur', 'Constructeur', 'Date','Designation', 'Entreprise', 'Etablissement', 'Immatriculation', 'LibelleCommercial', 'LibelleModele', 'Marque', 'MontantHT','Nature', 'NoLigne', 'NoOR', 'Nombredenregistrements', 'PGI','PrixUnitaire', 'Quantite', 'Reference', 'Temps', 'Type', 'VIN'])
#Veh['NoOR'] = Veh['NoOR'].astype(str)
len(Veh)
len(Veh.columns)

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

GTM = GTM[ [l.startswith("VF") for l in GTM["VIN"]] ]
## Ticket des vehicules dans GTM
Veh_IN_GTM = Veh[Veh['VIN'].isin(np.unique(GTM.VIN))]
Veh_IN_GTM = Veh_IN_GTM[ [l.startswith("VF") for l in Veh_IN_GTM["VIN"]] ]
# Et de ceux qui ne le sont pas
Veh_NOT = Veh[Veh['VIN'].isin(np.unique(GTM.VIN)) == False]
Veh_NOT = Veh_NOT[ [l.startswith("VF") for l in Veh_NOT["VIN"]] ]

######### ON GARDE LES COURBES QUASI PARFAITES ###################
len(np.unique(Veh_IN_GTM['VIN']))

P_Veh = Veh_IN_GTM.sort_values(by =['VIN','Compteur'])

P_Veh['KM_min'] = P_Veh.groupby('VIN')['Compteur'].transform(pd.Series.min)
P_Veh['KM_max'] = P_Veh.groupby('VIN')['Compteur'].transform(pd.Series.max)
P_Veh['cumsum'] = P_Veh.groupby('VIN')['Temps'].transform(pd.Series.cumsum)
P_Veh['Temps_max'] = P_Veh.groupby('VIN')['cumsum'].transform(pd.Series.max)
P_Veh = P_Veh[P_Veh['KM_max'] > 100000]
P_Veh = P_Veh[P_Veh['KM_min'] < 50000]
## Pas de nombre d'heures en atelier abérrant
P_Veh = P_Veh[P_Veh['Temps_max'] < 400]
#P_Veh = P_Veh[P_Veh['MontantHT'] > 0]


########### TEMPS ##########################################
# Stockage des resultats
Means_H = []
Medians_H = []
Res_H = []
lenG_H = []
P1_H =[]
P3_H = []
Nmax =  50

# par tranche de 10 000 km on calcul le nombre d'heures median des vehicules aux profils type
for i in range(0,Nmax):
    P_Veh_sub = P_Veh[P_Veh['Compteur'] > i*10000]
    P_Veh_sub = P_Veh_sub[P_Veh_sub['Compteur'] < (i+1)*10000]
    P_Veh_sub['SUM'] = P_Veh_sub.groupby('VIN')['Temps'].transform(pd.Series.sum)
    P_Veh_sub = P_Veh_sub.drop_duplicates(subset=['VIN'])
    lenG_H.append(len(P_Veh_sub))
    Means_H.append(np.mean(P_Veh_sub['SUM']))
    Medians_H.append(np.median(P_Veh_sub['SUM']))
    Res_H.append(P_Veh_sub['SUM'])
    P1_H.append(np.percentile(P_Veh_sub['SUM'], 25))
    P3_H.append(np.percentile(P_Veh_sub['SUM'], 75))
    #print(Means[i])

len(lenG_H)

#Means
#plt.scatter(np.linspace(0,Nmax-1,Nmax), Medians_H)
#plt.scatter(np.linspace(0,Nmax-1,Nmax), smooth(Medians_H,6))
################################################################

########### MontantHT ##########################################
Means = []
Medians = []
Res = []
lenG = []
P1 =[]
P3 = []
Nmax =  50

## Pareil que plus haut mais pour le CA
for i in range(0,Nmax):
    P_Veh_sub = P_Veh[P_Veh['Compteur'] > i*10000]
    P_Veh_sub = P_Veh_sub[P_Veh_sub['Compteur'] < (i+1)*10000]
    P_Veh_sub['SUM'] = P_Veh_sub.groupby('VIN')['MontantHT'].transform(pd.Series.sum)
    P_Veh_sub = P_Veh_sub.drop_duplicates(subset=['VIN'])
    lenG.append(len(P_Veh_sub))
    Means.append(np.mean(P_Veh_sub['SUM']))
    Medians.append(np.median(P_Veh_sub['SUM']))
    Res.append(P_Veh_sub['SUM'])
    P1.append(np.percentile(P_Veh_sub['SUM'], 25))
    P3.append(np.percentile(P_Veh_sub['SUM'], 75))
    #print(Means[i])

#############
##### Plot ##
#############

#data = []

#for i in range(0,50):
#    trace = go.Box(y=Res[i], showlegend=False, name = str(i), boxmean= True, marker = dict(opacity =0, color = 'rgb(107,174,214)'), line = dict(color = 'rgb(107,174,214)')  )
#    data.append(trace)

#data.append( go.Scatter( x = np.linspace(0,Nmax-1,Nmax).astype(int).astype(str),line = dict(color = ('rgb(205, 12, 24)'),width = 4), y = Medians, mode='lines', name='Medians' ) )
#data.append( go.Scatter( x = np.linspace(0,Nmax-1,Nmax).astype(int).astype(str),line = dict(color = ('rgb(0,100,0)'),width = 4), y = Means, mode='lines', name='Means' ) )

#py.iplot(data)

######### TEST sur un vehicule ################
H_opti = smooth(Medians_H,6) # On smooth la courbe des median mais on laisse la premiere valeur a l'originelle
H_opti[0] = Medians_H[0]

## Pour chaque VIN on calcul en fct de son km le nombre d'heures potentiel à lui facturer
TimeToAdd = []
VIN_LIST = []
for VIN_TEST in np.unique(Veh['VIN']):
    Sub_Veh = Veh[Veh['VIN'] == VIN_TEST]
    Sub_Veh = Sub_Veh[Sub_Veh['Compteur']<Nmax*10000]
    if math.isnan(np.max(Sub_Veh['Compteur'])):
        Tps_rest = 0
    else:
        kmin = math.floor(np.max(Sub_Veh['Compteur'])/10000)
        kmax = math.ceil(np.max(Sub_Veh['Compteur'])/10000)
        Sub_Veh = Sub_Veh[Sub_Veh['Compteur'] > kmin*10000 ]
        Sub_Veh = Sub_Veh[Sub_Veh['Compteur'] < kmax*10000]
        Tps_rest = H_opti[kmin] - np.sum(Sub_Veh['Temps'])
        if Tps_rest <0:
            Tps_rest = 0
        TimeToAdd.append(Tps_rest)
        VIN_LIST.append(VIN_TEST)

VIN_LIST = pd.DataFrame(VIN_LIST)
VIN_LIST['heures'] = TimeToAdd
VIN_LIST.to_csv(path_data + 'TimeToVIN.csv')
