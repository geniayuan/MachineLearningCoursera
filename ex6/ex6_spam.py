## Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import svm

from getVocabList import getVocabList
from processEmail import processEmail
from emailFeatures import emailFeatures
## Initialization
#clear ; close all; clc

# Load Vocabulary Dict to use
vocabDict, vocabList = getVocabList()

## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.
print('\nPreprocessing sample email (emailSample1.txt)...')

# Extract Features
file_contents = open('emailSample1.txt','r').readlines()
word_indices  = processEmail( ''.join(file_contents), vocabDict)

# Print Stats
#print("length", len(word_indices))
#print('Word Indices:', word_indices)

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n.
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.
#print('\nExtracting features from sample email (emailSample1.txt)......')

# Extract Features
#file_contents = readFile('emailSample1.txt');
#word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices, vocabDict)

# Print Stats
print('Length of feature vector: %d' % features.size)
print('Number of non-zero entries: %d' % sum(features==1))


## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
mat = scipy.io.loadmat('spamTrain.mat')
X = mat['X']
y = mat['y']

print('\nTraining Linear SVM (Spam Classification)...')
print('(this may take 1 to 2 minutes)...')

myC = 0.1
linear_svm = svm.SVC(C=myC, kernel='linear', tol=1e-3) # max_iter=200)
linear_svm.fit(X, y.flatten())
ptrain = linear_svm.score(X, y.flatten()) * 100
print('Training Accuracy: %f' % ptrain)
#ptrain2 = linear_svm.predict(X).reshape(-1,1)
#print('Training Accuracy: %f' % np.mean(ptrain2==y.reshape(-1,1)))

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
print('\nEvaluating the trained Linear SVM on a test set ...')

mat = scipy.io.loadmat('spamTest.mat')
Xtest = mat['Xtest']
ytest = mat['ytest']

ptest = linear_svm.score(Xtest, ytest.flatten()) * 100
print('Testing set Accuracy: %f' % ptest)
#ptest2 = linear_svm.predict(Xtest).reshape(-1,1)
#print('Testing set Accuracy: %f' % np.mean(ptest2==ytest.reshape(-1,1)))

## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#
weightsList = linear_svm.coef_[0,:].tolist()
indexList = sorted(range(len(weightsList)), key = lambda k: weightsList[k], reverse=True)
# Sort the weights and obtin the vocabulary list
#[weight, idx] = sort(model.w, 'descend')

print('\nMost import predictors of spam:')
for i in range(15):
    index = indexList[i]
    print('%-15s (%f)' % (vocabList[index], weightsList[index] ) )
print('\n')

print('\nLeast important predictors of spam:')
for i in range(15):
    index = indexList[-i-1]
    print('%-15s (%f)' % (vocabList[index], weightsList[index] ) )

#test_id = vocabDict['dollarnumb']-1
#print("weight:", vocabList[test_id], "," , weightsList[test_id])
'''
## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
#  The following code reads in one of these emails and then uses your
#  learned SVM classifier to determine whether the email is Spam or
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
filename = 'spamSample1.txt';

# Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
p = svmPredict(model, x);

print('\nProcessed #s\n\nSpam Classification: #d\n', filename, p);
print('(1 indicates spam, 0 indicates not spam)\n\n');
'''
