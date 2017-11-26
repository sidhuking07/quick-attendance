import numpy as np
import sys
import pickle
from sklearn import svm
import pandas as pd
import csv


measurements = pd.read_csv('/openface/testing-generated-embeddings/reps.csv', header=None)

with open("/openface/generated-embeddings/classifier.pkl", 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
        else:
                (le, clf) = pickle.load(f, encoding='latin1')

present = []

predicted = clf.predict_proba(measurements)

print predicted

for each in predicted:
    maxI = np.argmax(each)
    person = le.inverse_transform(maxI)
    confd = each[maxI]
    #print("Predict {} with {} confidence.".format(person.decode('utf-8'), confd))
    if confd>0.35:
    	present.append(person.decode('utf-8'))

present = set(present)
present = list(present)

np.savetxt("attendance_csv.csv", present, delimiter=",", fmt='%s')

print "These persons are present today : "
for each in present:
    print "--{}".format(each)
    
