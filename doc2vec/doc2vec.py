import pandas as pd
import gensim
import random
import logging
import collections
from sklearn.model_selection import train_test_split

with open('data.csv') as f: 
    data = pd.read_csv(f)
    data = data.iloc[:, 1:]

train, test = train_test_split(data, test_size=0.2)
train_noscores = train.iloc[:, 0]
test_noscores = test.iloc[:, 0]

def read_corpus(series, tokens_only = False): 
    for index, line in enumerate(series): 
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only: 
            yield tokens 
        else: 
            yield gensim.models.doc2vec.TaggedDocument(tokens, [index])

train_corpus = list(read_corpus(train_noscores))
test_corpus = list(read_corpus(test_noscores, tokens_only=True))

## train model with 50 dimensions and iterating over a training corpus 40 times;
## minimum word count is set to 2 to exclude very rare words
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count = 2, epochs = 40)
## vocabulary is essentially a list of all unique words extracte from the corpus
model.build_vocab(train_corpus)
print(f"Word 'трамп' appeared {model.wv.get_vecattr('трамп', 'count')} times in the training corpus.")
## train model
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
model.save('model_full_50_40')
## assess the model
ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)): 
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    second_ranks.append(sims[1])

counter = collections.Counter(ranks)
## test the model manually
doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

## Around 95% of times document is most similar to itself, which is expected.
## Moreover, some ranks are >2 therefrore model shows signs of not being overfit. 

## infer vectors for all train and test
train_vectors = []
test_vectors = []
# for doc in train_corpus: 
#     inferred_vector = model.infer_vector(doc.words)
#     train_vectors.append(inferred_vector)
train_vectors = [model.dv[i] for i in range(len(model.dv))]
train_vectors = pd.DataFrame(train_vectors)
train.reset_index(inplace=True, drop=True)
train_vectors['score']=train['score']
train_vectors.to_csv('train_vectors_full.csv', index = False)
for doc in test_corpus: 
    inferred_vector = model.infer_vector(doc)
    test_vectors.append(inferred_vector)
test_vectors = pd.DataFrame(test_vectors)
test.reset_index(inplace=True, drop=True)
test_vectors['score']=test['score']
test_vectors.to_csv('test_vectors_full.csv', index = False)
