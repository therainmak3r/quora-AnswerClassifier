## Submission for Quora Challenges: Answer Classifier
## Code by Utkarsh Sharma, IIT Delhi.
## email: sharma.utkarsh.iitd@gmail.com

import numpy
from sklearn.ensemble import RandomForestClassifier

input_file = open('input00.txt', 'r')
train_inputs = input_file.read()
train_inputs = [sentence.split() for sentence in train_inputs.split('\n')] # parse text file to break each sentence into words, after breaking entire string into sentences
train_inputs = filter(None, train_inputs) # remove empty arrays from array
number_train_inputs = train_inputs[0][0]

test_inputs = train_inputs[2 + int(number_train_inputs):] # select train and test data from appropriate range. +2 because labels for number of train and test data
test_inputs = [sentence[1:] for sentence in test_inputs] # only identifier, and not label is currently written in test_inputs, and so +1

train_inputs = train_inputs[1:1 + int(number_train_inputs)]
train_outputs = [int(sentence[1]) for sentence in train_inputs] # extract labels for training
train_inputs = [sentence[2:] for sentence in train_inputs] # remove answer identifier

output_file = open('output00.txt', 'r')
test_outputs = output_file.read()
test_outputs = [sentence.split() for sentence in test_outputs.split('\n')] # parse the text file to break into words
test_outputs = filter(None, test_outputs) # remove empty arrays from array

# remove the feature indices which are in the form 'feature_no:feature_value'
for i in xrange(0, len(train_inputs)):
	for j in xrange(0, len(train_inputs[i])):
		train_inputs[i][j] = float(train_inputs[i][j].split(':')[1])

for i in xrange(0, len(test_inputs)):
	for j in xrange(0, len(test_inputs[i])):
		test_inputs[i][j] = float(test_inputs[i][j].split(':')[1])

rf = RandomForestClassifier(n_estimators = 20, max_features = 'auto')
print '... training random forest'
rf.fit(train_inputs, train_outputs)

print '... testing'
confusion_matrix = numpy.zeros([2,2])
correct = 0
total = len(test_inputs)
for i in xrange(0, len(test_inputs)):
	prediction = rf.predict(test_inputs[i])
	prediction = int(prediction[0]) # converting prediction from numpy array of single value to python int
	true_label = int(test_outputs[i][1])
	
	if prediction == 1:
		prediction_print = '+1';
	else:
		prediction_print = '-1';
	print test_outputs[i][0], prediction_print
	
	if (prediction == true_label):
		correct += 1	
	if prediction == -1:
		if true_label == -1:
			confusion_matrix[0][0] += 1
		else:
			confusion_matrix[0][1] += 1
	else:
		if true_label == -1:
			confusion_matrix[1][0] += 1
		else:
			confusion_matrix[1][1] += 1
print confusion_matrix
print 'correct is ' , correct
print 'accuracy is ', correct/float(total)