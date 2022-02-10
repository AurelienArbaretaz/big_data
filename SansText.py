import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

firstdf = pd.read_csv("data.csv")

# Lets remove all rows with 0 ...
firstdf = firstdf[firstdf["PrixNuitee"] != 0]

# 0 -> 300
dfZeroToThreeHundred = firstdf[firstdf.PrixNuitee < 300]

dfFiltered = dfZeroToThreeHundred[['Longitude','Latitude','television_cable','wifi','seche_cheveux','monoxyde_carbone_detect','salle_sport','fer_repasser','parking_sur-place','extincteur','Climatisation','Jacuzzi','machine_laver','Capacite_accueil', 'NbChambres', 'frais_menage', 'Caution', 'frais_menage','seche_linge','cheminee_interieur','pourEnfants_famille','Piscine','NbLits','television']]
myPrixNuitee = dfZeroToThreeHundred['PrixNuitee']

X_train, X_test, y_train, y_test = train_test_split(dfFiltered, myPrixNuitee, test_size = 0.2, random_state=0)

X_test.to_csv("Xtest.csv", index=False)

LatEtLon = X_test.loc[:, ['Latitude', 'Longitude']]
 
X_test = X_test.drop(['Longitude', 'Latitude'], axis=1)
X_train = X_train.drop(['Longitude', 'Latitude'], axis=1)


model_bordeaux = Ridge(alpha=0.1)
model_bordeaux.fit(X_train, y_train)

y_predicted = model_bordeaux.predict(X_test)

X_test['PrixNuitee'] = y_predicted
X_test['Latitude'] = LatEtLon['Latitude']
X_test['Longitude'] = LatEtLon['Longitude']
X_test.to_csv("XtestAvecPrix.csv", index=False)