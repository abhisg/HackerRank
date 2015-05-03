import sys
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

lines = sys.stdin.readlines()
f = int(lines[0].strip().split()[0])
n = int(lines[0].strip().split()[1])

training_X = []
training_Y = []

for i in range(n):
	line = lines[i+1].strip().split()
	X=[]
	for i in range(len(line)-1):
		X.append(float(line[i]))
	training_X.append(X)
	training_Y.append(float(line[f]))

model = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear',linear_model.RidgeCV(alphas=np.arange(0,1,0.05),cv=5))])
model.fit(training_X,training_Y)

t = int(lines[n+1].strip())

test_X = []
test_Y = []
for i in xrange(t):
	line = lines[n+2+i].strip().split()
	X = []
	for i in range(len(line)):
		X.append(float(line[i]))
	test_X.append(X)

test_Y = model.predict(test_X).tolist()
for i in xrange(t):
	print test_Y[i]
