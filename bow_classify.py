import random as rn
import numpy as np
import pandas as pd
import os, json, sys, time, string
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import itertools

np.random.seed(1)
rn.seed(1)
wordVectorLength = 300

def findMetrics (cm):
    epslon = 1.0e-12
    metrics = {'tn' : int(cm[0,0]), 'fp' : int(cm[0,1]), 'fn' : int(cm[1,0]), 'tp' : int(cm[1,1])}
    metrics['precision'] = metrics['tp'] / (metrics['tp'] + metrics['fp'] + epslon )
    metrics['sensitivity'] = metrics['tp']/(metrics['tp'] + metrics['fn'] + epslon )  # recall
    metrics['specificity'] = metrics['tn'] /(metrics['tn'] + metrics['fp'] + epslon )
    metrics['accuracy'] = (metrics['tp'] + metrics['tn']) / (metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn'] + epslon )
    metrics['f1'] = 2.0 / (1.0/(metrics['sensitivity']+epslon) + 1.0/(metrics['precision'] + epslon))
    return metrics

def getEmbeddingMatrix (word_index):
    embedding_matrix = np.zeros((len(word_index), wordVectorLength))
    words = set(word_index.keys())
    f = open(os.environ["PRE_TRAINED_HOME"] + '/fasttext/crawl-300d-2M-subword.vec')
    goodCount = 0
    for line in f:
        values = line.split()
        word = values[0].strip()
        if (word in words):
            wv = np.asarray(values[1:], dtype='float32')
            if (len(wv) == wordVectorLength):
                goodCount = goodCount + 1
                embedding_matrix[word_index[word]] = wv
    zeroCount = len(words) - goodCount
    print ("# Total, Good, Zero Word Vectors, Source:", len(words), goodCount, zeroCount)
    f.close()
    return embedding_matrix

def sparseMultiply (sparseX, embedding_matrix):
    denseZ = []
    for row in sparseX:
        newRow = np.zeros(wordVectorLength)
        for nonzeroLocation, value in list(zip(row.indices, row.data)):
            newRow = newRow + value * embedding_matrix[nonzeroLocation]
        denseZ.append(newRow)
    denseZ = np.array([np.array(xi) for xi in denseZ])
    return denseZ

start_time = time.time()

args = sys.argv
if (len(args) < 5):
    print ("Need 2 arg... docrepo")
    sys.exit(0)
else:
    clf = args[1]
    docrepo = args[2]
    vectorsource = args[3]
    nwords = args[4]

f = open ("./data/" + docrepo + ".json",'r')
repoData = json.loads(f.read())
f.close()

docs, y = {}, {}
docs['train'], y['train'], docs['test'], y['test'], labelNames, max_seq_length, labelName2labelIndex, catCounts = repoData['train_docs'], np.array(repoData['train_labels']), repoData['test_docs'], np.array(repoData['test_labels']), np.array(repoData['labelNames']), repoData['max_seq_length'], repoData['labelName2LabelIndex'], repoData['catCounts']

maxWords = 0
for part in ['train', 'test']:
    if (nwords != 'full'):
        docs[part] = [x[0:int(nwords)] for x in docs[part]]
    maxWords = max(maxWords, max([len(doc) for doc in docs[part]]))
max_seq_length = maxWords

print ("Max Number of Words", maxWords)
print ('Types: train_labels, test_labels, labelNames:',type(y['train']), type(y['test']), type(labelNames))
print ('Shapes: train_labels, test_labels, labelNames:',y['train'].shape, y['test'].shape, labelNames.shape)
print ('labelName2labelIndex:',labelName2labelIndex)
print ('Category Counts:',catCounts)

X = docs['train'] + docs['test']
X=np.array([np.array(xi) for xi in X])          #   rows: Docs. columns: words
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1).fit(X)
word_index = vectorizer.vocabulary_
train_x = vectorizer.transform(np.array([np.array(xi) for xi in docs['train']]))
test_x = vectorizer.transform(np.array([np.array(xi) for xi in docs['test']]))
print ('Vocab Train Test {} {} {}'.format(len(word_index), str(train_x.shape), str(test_x.shape)))

if (vectorsource == 'fasttext'):
    embedding_matrix = getEmbeddingMatrix (word_index)
    train_x = sparseMultiply (train_x, embedding_matrix)
    test_x = sparseMultiply (test_x, embedding_matrix)
    print ('Dense Z: Train & Test {} {}'.format(str(train_x.shape), str(test_x.shape)))

load_time = time.time() - start_time
start_time = time.time()

if (clf == 'svm'):
    model = LinearSVC(tol=1.0e-6,max_iter=20000)
elif (clf == 'lr'):
    model = LogisticRegression(tol=1.0e-6,max_iter=20000)
metrics = {}
filename= "./results/" + nwords + "-" + clf + "-" + docrepo + "-" + vectorsource
if (docrepo == 'reuters'):
    classifier = OneVsRestClassifier(model)
    classifier.fit(train_x, y['train'])
    predicted = classifier.predict(test_x)
    elapsed_time = time.time() - start_time
    mcm = multilabel_confusion_matrix(y['test'], predicted)
    metricsByLabel = {}
    for i,label in enumerate(labelNames):
        metricsByLabel[label] = findMetrics(mcm[i])
        
    tcm = np.sum(mcm,axis=0)
    metrics['load_time'] = load_time        # seconds
    metrics['elapsed_time'] = elapsed_time        # seconds
    metrics['metrics_by_label'] = metricsByLabel
    metrics['total_metrics'] = findMetrics (tcm)
    metrics['multi_confusion_matrix'] = mcm.tolist()
    print ('Time Taken:', load_time, elapsed_time)
    print ('\nTotal Metrics:',metrics['total_metrics'])
else:
    train_y = [np.argmax(label) for label in y['train']]
    test_y = [np.argmax(label) for label in y['test']]
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    elapsed_time = time.time() - start_time
    cm = confusion_matrix(test_y, predicted)
    print (cm)
    metrics['load_time'] = load_time        # seconds
    metrics['elapsed_time'] = elapsed_time        # seconds
    metrics['confusion_matrix'] = cm.tolist()
    metrics['classification_report'] = classification_report(test_y, predicted, digits=4, target_names=labelNames, output_dict=True)
    metrics['total_metrics'] = metrics['classification_report']["weighted avg"]
    print (classification_report(test_y, predicted, digits=4, target_names=labelNames))
    print ('Time Taken:', load_time, elapsed_time)
    print ('\nTotal Metrics:',metrics['total_metrics'])

f = open ("./results/" + nwords + "-" + clf + "-" + docrepo + "-" + vectorsource + ".json",'w')
out = json.dumps(metrics, ensure_ascii=True)
f.write(out)
f.close()

