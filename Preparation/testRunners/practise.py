
import numpy as np 
import gensim
import time
	
def saveIt():
	outputfile = 'numbersTest.out'
	d1 = [1,2,3,4,5]
	d1 = np.asarray(d1)
	d2 = [7,8,9,10,11]
	d2 = np.asarray(d2)

	with open(outputfile, 'a') as writer:
		np.savetxt(writer, d1, fmt='%.8g', delimiter=",")

	with open(outputfile, 'a') as writer:
		np.savetxt(writer, d2, fmt='%.8g', delimiter=",")


def loadIt():
	stuff = 'numbersTest.out'
	array = np.loadtxt(stuff)
	print(array)

def getVec():
	begin = time.time()
	model = gensim.models.KeyedVectors.load_word2vec_format('/media/jwong/Transcend/GoogleNews-vectors-negative300.bin', binary=True)
	loaded = time.time()
	com = model.wv['computer']
	first = time.time()
	print(com.shape)
	king = model.wv['king']
	sec = time.time()
	print(king.shape)
	tgt = com + king
	aftersum = time.time()
	print(tgt.shape)

	print('Time taken to load: {}, Get computer: {}, Get king: {}, sum: {}'.format(loaded-begin, first-loaded, sec-first,aftersum-sec))

if __name__ == "__main__":
	jello = Ham()
	