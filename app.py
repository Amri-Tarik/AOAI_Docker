import pandas as pd
import numpy as np
import sys
from pickled import *

pca = unpickled("pca")
clf_elec = unpickled("clf_elec")
clf_indus = unpickled("clf_indus")
clf_meca = unpickled("clf_meca")

#replace with mongodb grabber /TODO
data = pd.read_csv('cleaned.csv')
data.drop(["Unnamed: 0","FILIERE","OUTPUT"],inplace=True,axis=1)
liste = data.iloc[int(sys.argv[1]),:]
liste = liste.values.reshape(1, -1)
###################################

X = pca.transform(liste)
print('{ "gi" : "' + str(float("{:.2f}".format(clf_indus.predict(X)[0]* 100))) + '", "gm" : "' + str(float("{:.2f}".format(clf_meca.predict(X)[0]* 100))) + '" , "ge" : "' + str(float("{:.2f}".format(clf_elec.predict(X)[0]* 100))) + '" }')