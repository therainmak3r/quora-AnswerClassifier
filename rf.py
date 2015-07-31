## Submission for Quora Challenges: Answer Classifier
## Code by Utkarsh Sharma, IIT Delhi.
## email: sharma.utkarsh.iitd@gmail.com

import numpy
from sklearn.ensemble import RandomForestClassifier
import sys

train_inputs = sys.stdin.readlines()
train_inputs = [sentence.split() for sentence in train_inputs] # parse text file to break each sentence into words, after breaking entire string into sentences
train_inputs = filter(None, train_inputs) # remove empty arrays from array
number_train_inputs = train_inputs[0][0]

test_inputs = train_inputs[2 + int(number_train_inputs):] # select train and test data from appropriate range. +2 because labels for number of train and test data
test_identifiers = [sentence[0] for sentence in test_inputs] # extract the identifiers
test_inputs = [sentence[1:] for sentence in test_inputs] # only identifier, and not output label is currently written in test_inputs, and so +1

train_inputs = train_inputs[1:1 + int(number_train_inputs)]
train_outputs = [int(sentence[1]) for sentence in train_inputs] # extract labels for training
train_inputs = [sentence[2:] for sentence in train_inputs] # remove answer identifier

# remove the feature indices which are in the form 'feature_no:feature_value'
for i in xrange(0, len(train_inputs)):
	for j in xrange(0, len(train_inputs[i])):
		train_inputs[i][j] = float(train_inputs[i][j].split(':')[1])

for i in xrange(0, len(test_inputs)):
	for j in xrange(0, len(test_inputs[i])):
		test_inputs[i][j] = float(test_inputs[i][j].split(':')[1])

rf = RandomForestClassifier(n_estimators = 20, max_features = 'auto')
# print '... training random forest'
rf.fit(train_inputs, train_outputs)

# print '... testing'
for i in xrange(0, len(test_inputs)):
	prediction = rf.predict(test_inputs[i])
	prediction = int(prediction[0]) # converting prediction from numpy array of single value to python int
	
	if prediction == 1:
		prediction_print = '+1';
	else:
		prediction_print = '-1'; 
	print test_identifiers[i], prediction_print
