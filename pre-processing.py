from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import numpy as np 
import tensorflow as tf 



def  load_vocabulary(vocab_file):
    vocabulary = []
    with codecs.getreader('utf-8')(tf.gfile.GFile(vocab_file,'rb')) as file:
        for word in file:
            vocabulary.append(word)
        
    return vocabulary



def create_embedding(embed_name, vocab_size, embed_size, dtype=tf.float32):
  """Create a new or load an existing embedding matrix."""
  
    embedding = tf.get_variable(
          embed_name, [vocab_size, embed_size], dtype)
  return embedding