import tensorflow as tf
from transformers import *
import random as rn
import numpy as np
from tqdm import tqdm
import os, json, sys, time
import time
import pandas as pd
from metrics import Metrics

start_time_0 = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

np.random.seed(1)
rn.seed(1)

print ('tensorflow.__version__:', tf.__version__)
print ('tensorflow.keras.__version__:', tf.keras.__version__)

args = sys.argv
if (len(args) < 5):
    print ("Need 4 args... which_bert, docrepo, # nwords, # epochs")
    sys.exit(0)
else:
    BERT_PATH = args[1]
    if (os.path.isdir(BERT_PATH)):
        from_pt = True
        bert_source = "local"
    else:
        from_pt = False
        bert_source = "s3"
    docrepo = args[2]
    nwords = args[3]
    epochs = int(args[4])
    if (docrepo == 'reuters'):
        multiLabel = True
        tfMetric = tf.keras.metrics.BinaryAccuracy()
        tfLoss = tf.losses.BinaryCrossentropy()
        activation = 'sigmoid'
        allMetrics = [tfMetric, tf.metrics.FalsePositives(), tf.metrics.FalseNegatives(), tf.metrics.TrueNegatives(), tf.metrics.TruePositives(), tf.metrics.Precision(), tf.metrics.Recall()]
    else:
        multiLabel = False
        tfMetric = tf.keras.metrics.CategoricalAccuracy()
        tfLoss = tf.losses.CategoricalCrossentropy()
        activation = 'softmax'
        allMetrics = [tfMetric]

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
max_seq_length = min(512, maxWords + 2)
print ("Max Number of Words", maxWords)
print ('Types: train_labels, test_labels, labelNames:',type(y['train']), type(y['test']), type(labelNames))
print ('Shapes: train_labels, test_labels, labelNames:',y['train'].shape, y['test'].shape, labelNames.shape)
print ('labelName2labelIndex:',labelName2labelIndex)
print ('Category Counts:',catCounts)

def prepareBertInput(tokenizer, docsChunk):
    idsChunk, masksChunk, segmentsChunk = [], [], []
    for doc in tqdm(docsChunk, desc="Converting docs to features"):
        tokens = tokenizer.tokenize(' '.join(doc))
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0 : (max_seq_length - 2)]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        masks = [1] * len(ids)
        # Zero-pad up to the sequence length.
        while len(ids) < max_seq_length:
            ids.append(0)
            masks.append(0)
        segments = [0] * max_seq_length
        assert len(ids) == max_seq_length
        assert len(masks) == max_seq_length
        assert len(segments) == max_seq_length
        idsChunk.append(ids)
        masksChunk.append(masks)
        segmentsChunk.append(segments)
    encodedChunk = [idsChunk, masksChunk, segmentsChunk]
    return encodedChunk

encoded = {}
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
for chunk in ['train', 'test']:
    print ('BERT. Working on encoding ', chunk, ' docs')
    encoded[chunk] = prepareBertInput(tokenizer, docs[chunk])
    input_ids, input_mask, segment_ids = [tf.keras.backend.cast(x, dtype="int32") for x in encoded[chunk]]
    encoded[chunk] = [input_ids, input_mask, segment_ids]

def getModel():
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name="bert_input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name="bert_input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name="bert_segment_ids")
    inputs = [in_id, in_mask, in_segment]
#    scores = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=False, output_hidden_states=False)(inputs)
    top_layer_vectors, top_cls_token_vector = TFBertModel.from_pretrained(BERT_PATH, from_pt=from_pt)(inputs)
    predictions = tf.keras.layers.Dense(len(labelNames), activation=activation,use_bias=False)(top_cls_token_vector)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0), loss=tfLoss, metrics=allMetrics)
    print(model.summary())
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='results/bert.png')
    return model

model = getModel()
start_time = time.time()
history = model.fit(encoded['train'], y['train'], verbose=2, shuffle=True, epochs=epochs, batch_size=32)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
f = open ("results/" + nwords + '-history-' + bert_source + '-bert-' + docrepo + '.json', 'w')
f.write(hist.to_json())
f.close()

evalInfo = model.evaluate(encoded['test'], y['test'], verbose=2)
print ('Test Metrics:', model.metrics_names)
print ('Test Evaluation:', evalInfo)

predicted = model.predict(encoded['test'], verbose=2)
elapsed_time = time.time() - start_time
print ('Time Taken:', elapsed_time)

metrics = Metrics().computeMetrics (labelNames, labelName2labelIndex, docs['test'], y['test'], predicted, multiLabel)
print ('\nComputed Metrics from model.predict:',metrics['totalFtCounts'])
metrics['Test_Eval'] = {}
for i, item in enumerate(model.metrics_names):
    metrics['Test_Eval'][item] = float(evalInfo[i])
f = open ("results/" + nwords + '-' + bert_source + '-bert-' + docrepo + '.json', 'w')
out = json.dumps(metrics, ensure_ascii=True)
f.write(out)
f.close()

