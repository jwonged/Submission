'''
Created on 3 Jan 2018

@author: jwong
'''
import pickle

from InputReader import InputReader
import numpy as np
import tensorflow as tf 


def test():
    filePath = '/media/jwong/Transcend/GoogleNews-vectors-negative300.txt'
    with open(filePath) as file:
        embeddings
        L = tf.Variable(embeddings, dtype=tf.float32, trainable=False)
        # shape = (batch, sentence, word_vector_size)
        pretrained_embeddings = tf.nn.embedding_lookup(L, word_ids)

def main():
    dummySetDirX = '/media/jwong/Transcend/VQADataset/DummySets/DummySparseCleanWVsum1000Trainx.pkl'
    dummySetDirY = '/media/jwong/Transcend/VQADataset/DummySets/DummySparseCleanWVsum1000Trainy.pkl'
    trainReader = InputReader(dummySetDirX, dummySetDirY)

if __name__ == '__main__':
    test()