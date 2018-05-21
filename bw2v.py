
import numpy as np
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import codecs
import collections
import math

import tensorflow as tf

prop = fm.FontProperties(fname='kalpurush.ttf')

def  create_vocabulary(file_name,num_words=2500):
    vocabulary = []
    with codecs.getreader('utf-8')(tf.gfile.GFile(file_name,'rb')) as file:
        words = ' '.join(file).split()
        counter=[['UNK',-1]]
        counter.extend(collections.Counter(words).most_common(num_words))
        unique_words = [word[0] for word in counter]
        dictionary = {w:i for i,w in enumerate(unique_words)}
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        data=[]
        unk_count = 0
        for word in words:
            index = dictionary.get(word, 0)
            if index == 0:  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        counter[0][1] = unk_count

    return data,reversed_dictionary, len(dictionary)
 

data,reverse_dictionary, vocab_size= create_vocabulary('data.txt')

#print(data[:10])

WINDOW_SIZE = 2
dataset = []
for i ,word in enumerate(data):
        for nb_word in data[max(i - WINDOW_SIZE, 0) : min(i + WINDOW_SIZE, len(data)) + 1] : 
            if nb_word != word:
                dataset.append([word, nb_word])
print(dataset[:5])

def batify(size):
    assert size < len(dataset)
    source = []
    target = []
    random_data = np.random.choice(range(len(dataset)),size,replace=False)
    for item in random_data:
        source.append(dataset[item][0])
        target.append([dataset[item][1]])
    return source, target

#print(batify(3))

batch_size = 128
EMBEDDING_DIM = 32 
SAMPLE_SIZE = 128
# making placeholders for x_train and y_train
x = tf.placeholder(tf.int32, shape=[batch_size])
y_label = tf.placeholder(tf.int32, shape=[batch_size,1])

with tf.device('/cpu:0'):
    embedding_weights = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_DIM],-1.0,1.0))
    embed = tf.nn.embedding_lookup(embedding_weights,x)

nce_weights = tf.Variable(
  tf.truncated_normal([vocab_size, EMBEDDING_DIM],
                      stddev=1.0 / math.sqrt(EMBEDDING_DIM)))
nce_biases = tf.Variable(tf.zeros([vocab_size]))
cross_entropy_loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights,nce_biases,y_label,embed,SAMPLE_SIZE,vocab_size))

train_step = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy_loss)
norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_weights), 1, keep_dims=True))
normalized_embeddings = embedding_weights / norm
  
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) 

n_iters = 1000

for i in range(n_iters):
    x_data, y_data = batify(batch_size)
    sess.run(train_step, feed_dict={x: x_data, y_label: y_data})
    if i%1000==0:
        print('loss after %d steps: '%(i), sess.run(cross_entropy_loss, feed_dict={x: x_data, y_label: y_data}))

embeddings = sess.run(normalized_embeddings)

fig, ax = plt.subplots()
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 20])
plot_only = 250

tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
  
for i,word in enumerate(labels):
    x,y = low_dim_embs[i,:]
    ax.scatter(x,y)
    ax.annotate(word, (x,y),fontproperties=prop)
plt.show()