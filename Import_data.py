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

plotly.tools.set_credentials_file(username='valentin.lefranc', api_key='Xb0HU4LGnX8h3COhUgJr')

path_g = "C:/Users/Lefranc/Documents/Dossier_Client_Temporaire/BernisInvestissement2017-6BT-GTM/"
path_data = path_g + "donnees/"
path_prog = path_g + "programmes/"

exec(open(path_prog+'Helper.py').read())

GTM = pd.read_csv(path_data + "GTM-ListeMVSeed_T.csv",sep=';',decimal=b',')
Veh = pd.read_csv(path_data + "Vehicules-HistoriqueSeed_T.csv",sep=';',decimal=b',')

Veh['VIN'] = Veh['VIN'].astype(str)
GTM['VIN'] = GTM['VIN'].astype(str)

Veh.columns = np.array(['CodeClient', 'Compteur2', 'Compteur', 'Constructeur', 'Date','Designation', 'Entreprise', 'Etablissement', 'Immatriculation', 'LibelleCommercial', 'LibelleModele', 'Marque', 'MontantHT','Nature', 'NoLigne', 'NoOR', 'Nombredenregistrements', 'PGI','PrixUnitaire', 'Quantite', 'Reference', 'Temps', 'Type', 'VIN'])

# On ne garde que les lignes pour lesquelles VIN commence par VF
Veh = Veh[ [l.startswith("VF") for l in Veh["VIN"]] ]


# nb d'occurence de chaque code VIN
List_VIN = Counter(Veh['VIN'])
type(List_VIN)
List_VIN = pd.DataFrame.from_dict(List_VIN, orient='index').reset_index()
List_VIN.columns = np.array(['type','oc'])
List_VIN = List_VIN.sort_values(by =['oc'], ascending=False)
List_VIN



# On isole le vehicule avec le plus d'OR
Veh_OR_max = Veh[ Veh['VIN'] == "VF622ACB000101129"]
# Cout total des OR
sum(Veh_OR_max['MontantHT'])

# Nombre d'OR
len(Veh_OR_max)
N_OR = Veh_OR_max.groupby(['Date']).apply(len)
N_OR = pd.DataFrame(N_OR)
N_OR['Date'] = N_OR.index
N_OR.columns = np.array(['N_OR','Date'])
N_OR['Date'] = pd.to_datetime(N_OR['Date'], format="%d/%m/%Y")
N_OR = N_OR.sort_values(by =['Date'], ascending=False)
N_OR
a =Veh[ Veh['Date'] == '31/10/2013']
a = a[a['VIN'] == "VF622ACB000101129"]
a['Designation']


Veh[ Veh['VIN'] == "VF644AGL000004421"]


len(Veh_OR_max[Veh_OR_max['Date'] == '02/05/2011'])


############### Creation de la liste de stop word ###############
FR_stop = stopwords.words('French')
FR_stop_upper = [ w.upper() for w in FR_stop]
FR_stop = FR_stop + FR_stop_upper
FR_stop = [ insert_sting_middle(' ',w) for w in FR_stop]
FR_stop = FR_stop + ["d'","D'","c'","C'","qu'","QU'","l'","L'"]
FR_stop = np.array(FR_stop)
###############################################################

####### On enleve les stop words de Designation################
Des = list(Veh_OR_max['Designation'])
Des = [ str(w) for w in Des]
for i in range(0,len(FR_stop)):
    Des = [ re.sub(FR_stop[i], '',w) for w in Des]

Des = np.array(Des)
Veh_OR_max['Designation_stop'] = Des
#############################################################
##############################################################

GTM.columns = np.array(['Action', 'AnneePrevisionnelle', 'ClientFacturation', 'Client',
  'CodeAction', 'CodeEntrepriseEntretien',
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

######### Liste des actions de maintenance #########
ActGTM = list(GTM['DesignationAction'])
ActGTM = [ str(w) for w in ActGTM]
ActGTM = list(set(ActGTM))
ActGTM = ActGTM + [ w.lower() for w in ActGTM]
ActGTM = np.array(ActGTM)
##################################################

D_stop = list(Veh_OR_max['Designation_stop'])

#### Matrice pour stocker les résultats
S = (len(ActGTM),len(D_stop))
Sim_Matrix = np.zeros(S)

for i in range(0,len(ActGTM)):
    Sim_Matrix[i] = [similar(ActGTM[i],w) for w in D_stop]

### Pourl'instant on fixe 0.7 pour tout et on garde comme oppération référencée unique si la distance sup à 0.7
Sim_Matrix = pd.DataFrame(Sim_Matrix)
Sim_Matrix = Sim_Matrix > 0.7

Sim_OR = Sim_Matrix.sum(axis =0)
#np.max(Sim_OR)
#l = plt.plot(Sim_OR)

######### On isole les cas des presta périodiques et on extrait le OR correspondant
Veh_OR_max["OP_refer"] = np.array(Sim_OR)
Veh_max_OP = Veh_OR_max[Veh_OR_max['OP_refer']>0]
Veh_max_OP = Veh_max_OP[Veh_max_OP['Designation_stop'] != "nan"]
Veh_OR_Presta = Veh_max_OP[['Date','Designation_stop','NoOR']]

Veh_OR_Presta = Veh_OR_Presta.drop_duplicates(subset = ['NoOR'])

OR_Presta = np.unique(np.array(Veh_OR_Presta['NoOR'])).astype(int)
Des_Presta = np.array(Veh_OR_Presta['Designation_stop']).astype(str)


for i in range(0,len(OR_Presta)):
    Veh_OR = Veh_OR_max[Veh_OR_max['NoOR'] == OR_Presta[i]]
    print(Veh_OR['Designation_stop'])

Veh_OR['Designation_stop']

len(OR_Presta)


N_OP_max = max(Veh_OR_max.groupby(['NoOR']).apply(len))

Asso_Presta = np.zeros((len(OR_Presta),N_OP_max))
Asso_Presta

Asso_Presta = np.array(Asso_Presta)

Veh_OR = Veh_OR_max[Veh_OR_max['NoOR'] == OR_Presta[i]]
presta = np.append( Des_Presta[i] , Veh_OR['Designation_stop'])
presta = np.append(presta,np.zeros(N_OP_max-len(presta)))
Asso_Presta[i] = presta
    #presta = np.append( Des_Presta[i] , Veh_OR['Designation_stop'])
    #presta = np.append(presta,np)
    #Asso_Presta =
