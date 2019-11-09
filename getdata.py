import numpy as np
import string, json, sys, os
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import reuters 

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk_stopw = stopwords.words('english')

use_seq_length = 1000000000

def tokenize (text):        #   no punctuation & starts with a letter & between 2-15 characters in length
    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return  [f.lower() for f in tokens if f]

def get20News():
    twenty_news = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
    docs, labels, labelIndexToLabelName, allWords, docLengthCounts, catCounts, max_sequence_length = [], [], {}, set(), [], {}, 0
    for i, article in enumerate(twenty_news['data']):
        tokens = tokenize (article)
        docLength = len(tokens)
        if (docLength > 0):
            docLengthCounts.append(docLength)
            max_sequence_length = max(max_sequence_length, docLength)
            if (docLength > use_seq_length):
                doc = tokens[0:use_seq_length]
            else:
                doc = tokens.copy()

            docs.append(doc)
            label = [0]*20
            groupIndex = twenty_news['target'][i]
            groupName = twenty_news['target_names'][groupIndex]
            label[groupIndex] = 1
            if groupName in catCounts:
                catCounts[groupName] = catCounts[groupName] + 1
            else:
                catCounts[groupName] = 1
            allWords.update(set(doc))
            labels.append(label)
            labelIndexToLabelName[groupIndex] = groupName

    if (len(catCounts) != 20):
        print('Error...')
        sys.exit()
    plotFig('20news', catCounts, docLengthCounts)

    labelIndexToLabelNameSortedByIndex = sorted(labelIndexToLabelName.items(), key=lambda kv: kv[0]) # List of tuples sorted by the label number [ (0, ''), (1, ''), .. ]
    labelNamesInIndexOrder = [item[1] for item in labelIndexToLabelNameSortedByIndex]
    labelName2LabelIndex = {v:int(k) for k, v in labelIndexToLabelName.items()}
    train_docs, test_docs, train_labels, test_labels = train_test_split(docs, labels, test_size=0.20, random_state=1)

    printStats (len(labelNamesInIndexOrder), labelNamesInIndexOrder, labelName2LabelIndex, catCounts, train_docs, train_labels, test_docs, test_labels, docLengthCounts,max_sequence_length, len(allWords))

    result = {'train_docs': train_docs, 'train_labels' : train_labels, 'test_docs': test_docs, 'test_labels' : test_labels, 'labelNames' : labelNamesInIndexOrder, 'max_seq_length' : use_seq_length, 'labelName2LabelIndex' : labelName2LabelIndex, 'catCounts' : catCounts}

    f = open ('data/20news.json','w')
    out = json.dumps(result, ensure_ascii=True)
    f.write(out)
    f.close()

def getMovies():
    docs, labels, label, allWords, docLengthCounts, max_sequence_length, nCats = {}, {}, {}, set(), [], 0, 2
    labelIndexToLabelName = {0 : 'neg', 1 : 'pos'}
    labelIndexToLabelNameSortedByIndex = [(0, 'neg'), (1, 'pos')]
    catCounts = {'neg' : 0, 'pos' : 0}
    label['neg'] = [1, 0]
    label['pos'] = [0, 1]
    for dataset in ['train', 'test']:
        docs[dataset], labels[dataset] = [], []
        for directory in ['neg', 'pos']:
            dirName = './data/aclImdb/' + dataset + "/" + directory
            for reviewFile in os.listdir(dirName):
                catCounts[directory] = catCounts[directory] + 1
                with open (dirName + '/' + reviewFile, 'r') as f:
                    article = f.read()
                    tokens = tokenize (article)
                    docLength = len(tokens)
                    if (docLength > 0):
                        docLengthCounts.append(docLength)
                        max_sequence_length = max(max_sequence_length, docLength)
                        if (docLength > use_seq_length):
                            doc = tokens[0:use_seq_length]
                        else:
                            doc = tokens.copy()
                        allWords.update(set(doc))
                        docs[dataset].append(doc)
                        labels[dataset].append(label[directory])
    plotFig('movies', catCounts, docLengthCounts)

    labelNamesInIndexOrder = [item[1] for item in labelIndexToLabelNameSortedByIndex]
    labelName2LabelIndex = {v:int(k) for k, v in labelIndexToLabelName.items()}

    printStats (nCats, labelNamesInIndexOrder, labelName2LabelIndex, catCounts, docs['train'], labels['train'], docs['test'], labels['test'], docLengthCounts, max_sequence_length, len(allWords))

    result = {'train_docs': docs['train'], 'train_labels' : labels['train'], 'test_docs': docs['test'], 'test_labels' : labels['test'], 'labelNames' : labelNamesInIndexOrder, 'max_seq_length' : use_seq_length, 'labelName2LabelIndex' : labelName2LabelIndex, 'catCounts' : catCounts}

    f = open ('data/movies.json','w')
    out = json.dumps(result, ensure_ascii=True)
    f.write(out)
    f.close()
    plotFig('movies', catCounts, docLengthCounts)

def getReuters():
    train_docs, train_labels, test_docs, test_labels, allWords, docLengthCounts, max_sequence_length  = [], [], [], [], set(), [], 0
    labelNamesInIndexOrder = reuters.categories()
    labelNamesInIndexOrder.sort()
    nCats = len(labelNamesInIndexOrder)
    labelName2LabelIndex = dict(zip(labelNamesInIndexOrder,range(0,nCats)))
    for doc_id in reuters.fileids():
        doc0 = tokenize(reuters.raw(doc_id))
        docLength = len(doc0)
        max_sequence_length = max(max_sequence_length, docLength)
        docLengthCounts.append(docLength)

        if (docLength > use_seq_length):
            doc = doc0[0:use_seq_length]
        else:
            doc = doc0.copy()

        allWords.update(set(doc))
        cats = reuters.categories(doc_id)
        labels = np.zeros(nCats, dtype='int')
        for cat in cats:
            labels[labelName2LabelIndex[cat]] = 1
        if doc_id.startswith("train"):		
            train_docs.append(doc)
            train_labels.append(labels.tolist())
        else:
            test_docs.append(doc)
            test_labels.append(labels.tolist())

    catCounts = {}
    for cat in labelNamesInIndexOrder:
        catCounts[cat] = len(reuters.fileids(cat))

    printStats (nCats, labelNamesInIndexOrder, labelName2LabelIndex, catCounts, train_docs, train_labels, test_docs, test_labels, docLengthCounts, max_sequence_length, len(allWords))

    result = {'train_docs': train_docs, 'train_labels' : train_labels, 'test_docs': test_docs, 'test_labels' : test_labels, 'labelNames' : labelNamesInIndexOrder, 'max_seq_length' : use_seq_length, 'labelName2LabelIndex' : labelName2LabelIndex, 'catCounts' : catCounts}

    f = open ('data/reuters.json','w')
    out = json.dumps(result, ensure_ascii=True)
    f.write(out)
    f.close()
    plotFig('reuters', catCounts, docLengthCounts)

def printStats(nCats, allCats, labelName2LabelIndex, catCounts, train_docs, train_labels, test_docs, test_labels, docLengthCounts, max_sequence_length, vocab_size):
    print ('# All Docs:', len(docLengthCounts))
    print ('# Cats:', nCats)
    print ('Categories:', allCats)
    print ('labelName2LabelIndex:', labelName2LabelIndex)
    print ('catCounts:', catCounts)
    print ('Used Docs/Labels & Train Docs/Labels & Test Docs/Labels:', len(train_docs)+len(test_docs), '/', len(train_labels)+len(test_labels), len(train_docs),'/',len(train_labels), len(test_docs),'/',len(test_labels))
    print ('# docs <= 510 tokens:',len([i for i in docLengthCounts if i <= 510]))
    print ('# docs <= 500 tokens:',len([i for i in docLengthCounts if i <= 500]))
    print ('# docs <= 400 tokens:',len([i for i in docLengthCounts if i <= 400]))
    print ('# docs <= 300 tokens:',len([i for i in docLengthCounts if i <= 300]))
    print ('# docs <= 200 tokens:',len([i for i in docLengthCounts if i <= 200]))
    print ('# docs <= 175 tokens:',len([i for i in docLengthCounts if i <= 175]))
    print ('# docs <= 100 tokens:',len([i for i in docLengthCounts if i <= 100]))
    print ('max_sequence_length:', max_sequence_length)
    print ('Vocab Size:', vocab_size)

def plotFig (filename, catCounts, docLengthCounts):
    fig, ax = plt.subplots()
    yVals = list(catCounts.values())
    xVals = range(0,len(yVals))
    plt.bar(xVals, yVals)
    plt.tight_layout()
    fig.savefig('data/' + filename + '-catCounts.png', format='png', dpi=720)
    plt.close(fig)
    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(docLengthCounts, range=[0, 250], bins=50)
    fig.savefig('data/' + filename + '-docLengths.png', format='png', dpi=720)
    plt.close(fig)

print ('\nWorking on movies data...\n')
getMovies()

print ('\nWorking on reuters data...\n')
getReuters()

print ('\nWorking on 20news data...\n')
get20News()

