import numpy as np
import pandas as pd
from datetime import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def create_dataset(nom_action : str, start_date : str, end_date : str, nb_jour : int) :
    '''
        nom_action : Il s'agit du nom de l'action que vous voulez récupérer
            Exemple : AAPL (Apple) AMZN (Amazon) FB (Facebook)
        start_date : Date de début pour récupérer les données
        end_date : Date de fin pour récupérer les données
        nb_jour : Le nombre de jour que nous allons prendre en compte pour la prédiction
    '''
    df = web.DataReader(nom_action,"yahoo", start_date,end_date) #Recupère les actions depuis Yahoo Finance

    df_jour = pd.DataFrame(index=df.index)
    df_jour["Today"] = df["Adj Close"]
    df_jour["Volume"] = df["Volume"]

    for i in range(0,nb_jour):
        df_jour["Jour%s" % str(i+1)] = df["Adj Close"].shift(i+1)

    dfret = pd.DataFrame(index=df_jour.index)
    dfret["Volume"] = df_jour["Volume"]
    dfret["Today"] = df_jour["Today"].pct_change()*100.0 #Nous permet de calculer la difference en pourcentage d'un jour à l'autre
    
    '''
        Le pct_change permet de connaitre la différence entre 2 valeurs
            Exemple :
                2021-01-04  196.059998
                2021-01-05  202.929993

                Entre ceux deux valeurs, nous avons une différence de +3.5%
    '''
    
    for i in range(0,nb_jour):
        dfret["Jour%s" % str(i+1)] = df_jour["Jour%s" % str(i+1)].pct_change()*100.0

    dfret["Evolution"] = np.sign(dfret["Today"])

    dfret.drop(dfret.index[:nb_jour+1], inplace=True) #On retire les (nb_jour+1) premier car nous n'avons pas la valeur pour le jour ie NaN

    return dfret

if __name__ == "__main__":

    data = create_dataset("SE", datetime(2021,1,1), datetime(2022,2,12),nb_jour=4) #Création du dataset

    X = data[["Jour1","Jour2","Jour3","Jour4"]]
    y = data["Evolution"]

    start_test = datetime(2022,1,1)

    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]

    X_test = X[X.index >= start_test]
    y_test = y[y.index >= start_test]

    model = KNeighborsClassifier(150)
    fittedModel = model.fit(X_train,y_train)

    pred = model.predict(X_test)

    print("Accuracy : ", model.score(X_test,y_test))
    cm = confusion_matrix(y_test, pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()

    plt.show()