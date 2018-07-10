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
path_data = path_g + "/Donnees/"
path_prog = path_g + "/programmes/"

exec(open(path_prog+'Helper.py').read())

Veh = pd.read_pickle(path_data + 'VehSimMatrix.pkl')


Veh['VIN'] = Veh['VIN'].astype(str)

######### Liste des actions de maintenance #########
ActGTM = Veh.columns[24:]
ActGTM = [ str(w) for w in ActGTM]
ActGTM = list(set(ActGTM))
#ActGTM = np.array(ActGTM)
# On selectionne une presta en particulier ######
#ActGTM
#ActGTM = ActGTM[4]
#ActGTM_str = re.sub(' ', '',str(ActGTM))
##################################################
#Veh.columns[24:]

first = 0

for i in range(0,len(ActGTM)):
    print('#####################')
    print(ActGTM[i])
    print('#####################')
    Veh_VM = Veh[Veh[ActGTM[i]] > 0.7]
    print('Nb apparition de l ACM  : ' ,len(Veh_VM))
    if len(Veh_VM) > 0:
        ## Liste des OR associé à une ActGTM
        print('Liste OR associees')
        Veh_VM_OR = Veh_VM['NoOR']
        Veh_VM_Des = Veh[Veh['NoOR'].isin(Veh_VM_OR)]
        Veh_VM_Des = Veh_VM_Des[Veh_VM_Des[ActGTM[i]] < 0.7]

        # On cherche les longueurs des chaine de characteres et on enleve les trop courtes ##
        print("On enleve les chaines de char trop courtes")
        Des_VM = list(Veh_VM_Des['Designation'])
        Des_VM = [str(w) for w in Des_VM]
        Len_Des_VM = [len(w) for w in Des_VM]
        Des_Len = np.array(Len_Des_VM)
        Des_Len = pd.DataFrame(Des_Len)
        Des_Len.columns = np.array(["Len"])
        Des_Len['Des_VM'] = Des_VM
        Des_Len = Des_Len[Des_Len['Len'] > 5]


        #####################################################################################

        ###### On cherche ensuite les chaines les plus récurentes
        print('On cherche les chaines les plus présentes')
        Des_Len2 = Des_Len.groupby('Des_VM').apply(len)
        Des_Len2 = pd.DataFrame(Des_Len2)
        Des_Len2['Des_Len'] = Des_Len2.index
        Des_Len2.columns = np.array(['no','Des_Len'])
        Des_Len2 = Des_Len2.sort_values(by =['no'], ascending=False)

        #Pour eviter les erreurs de memoire
        if len(Des_Len2) > 10000:
            Des_Len2 = Des_Len2[Des_Len2['no'] > 1]


        ######### On va ensuite corriger pour rassembler les chaines quasi similaires ########
        print('correction des chaines les plus proche')
        S = (0,len(Des_Len2))
        Sim_Matrix = np.zeros(S)
        Des_Len2 = list(Des_Len2['Des_Len'])
        Sim_Matrix = pd.DataFrame(Sim_Matrix)

        ###### On créé une matrice nxn donnant les correspondance de chaque phrase
        print('creation de la matrice // Taille :')
        print(len(Des_Len2))

        list_to_drop =  []

        print('creation de la matrice')
        for k in range(0,len(Des_Len2)):
            Sim = [text_cosine(Des_Len2[k],w) for w in Des_Len2]
            Sim = np.array(Sim)
            taille = np.where(np.logical_and(Sim>=0.9, Sim<1))
            #print(str(k) + "   " + str(taille[0]))
            if len(taille[0]) == 0:
                list_to_drop.extend([k])
            if len(taille[0]) > 0:
                Sim_Matrix = Sim_Matrix.append(pd.DataFrame(Sim).T)

        len(list_to_drop)
        Sim_Matrix = Sim_Matrix.drop(Sim_Matrix.columns[list_to_drop],axis = 1)

        Des_Len2=np.array(Des_Len2)

        Sim_Matrix = pd.DataFrame(Sim_Matrix)
        print(len(list_to_drop))
        print(Sim_Matrix.shape)
        print(len(np.delete(Des_Len2,list_to_drop)))
        Sim_Matrix.index = np.delete(Des_Len2,list_to_drop)
        Sim_Matrix.columns = np.delete(Des_Len2,list_to_drop)



        Switch_init = np.array([])
        Switch_tc = np.array([])
        avoid = np.array([])
        jmax = np.linspace(0,len(Sim_Matrix),len(Sim_Matrix)+1).astype(int)
        print("tableau de switch")
        for k in range(0,len(Sim_Matrix)):
            for j in jmax-1:
                if Sim_Matrix.iloc[k][j] > 0.9 and k !=j :
                    Switch_tc = np.append(Switch_tc,Sim_Matrix.columns[j])
                    Switch_init = np.append(Switch_init,Sim_Matrix.index[k])
                    avoid = np.append(avoid,j)
                    #print(Sim_Matrix.index[i], '//////' ,Sim_Matrix.index[j] )
        ###############################################################################

        #### Tableau de correspondance
        print('tableau de correspondance')
        Switch_tc = pd.DataFrame(Switch_tc)
        Switch_tc['Switch_init'] = Switch_init
        Switch_tc.columns = np.array(['tc','init'])

        Sw2 = Switch_tc
        rm = np.array([])

        for k in range(0,len(Switch_tc)):
            if np.isin(Switch_tc['tc'][k],Switch_tc['init'][0:k]):
                rm = np.append(rm,k)

        ####### On enleve les doubles correspondance
        print('on enleve les doubles')
        Sw2 = Sw2.drop(rm)
        Sw2 = Sw2.drop_duplicates(subset=['tc'], keep="first")
        #### On remplace dans le array des Designations ########
        Des_Len_temp = np.array(Des_Len['Des_VM'])


        for k in range(0,len(Des_Len_temp)):
            if np.isin(Des_Len_temp[k],Sw2['tc']) :
                val, = np.where(Sw2['tc'] == Des_Len_temp[k])
                #print(Des_Len_temp[i],' // ',np.array(Sw2['init'])[int(val)],' // ',val, ' // ', i)
                Des_Len_temp[k] = np.array(Sw2['init'])[int(val)]

        ### On trouve enfin les prestas les plus souvent associees a l'originale ###################
        unique_elements, counts_elements = np.unique(Des_Len_temp, return_counts=True)
        #print('presta associees a : ' + ActGTM[i])
        res = pd.DataFrame()
        res['unique_elements'] = unique_elements
        res['counts_elements'] = counts_elements
        res = res.sort_values(by =['counts_elements'], ascending=False)
        print(res.head(25))
        titre = ActGTM[i]
        res['ActGTM'] = titre
        res.columns = np.array([titre,'counts','ActGTM'])
        # check if it is the first time in the loop
        if first == 0 :
            res_tot = res
            first = first + 1
        else:
            res_tot = res_tot.append(res)
            first = first + 1
        print('Nb = ',str(first) , 'Taille du fichier = ' + str(len(res_tot)) )

res_tot.to_pickle(path_data + 'Presta_Asso.pkl')
