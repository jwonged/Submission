'''
Created on 30 Mar 2018

@author: jwong
'''
import shelve
import pickle
from nltk import word_tokenize
from utils.Input_Processor import InputProcessor

class OnlineProcessor(InputProcessor):
    """
    For processing input from different sources in a web app
    """
    
    def __init__(self, imgFile, config):
        super(OnlineProcessor, self).__init__(config)
        
        print('Reading {}'.format(imgFile))
        self.imgData = shelve.open(imgFile, flag='r', protocol=pickle.HIGHEST_PROTOCOL)
        
        self.config  = config
        self.classToAnsMap = config.classToAnsMap
        self.classToAnsMap[-1] = -1
        self.mapWordToID = config.mapWordToID
    
    def processInput(self, qn, img_id):
        qnAsWordIDs = self._mapQnToIDs(qn)
        img_vec = self.imgData[img_id]
        batchOfQnsAsWordIDs, qnLengths = self._padQuestionIDs([qnAsWordIDs], 0)
        return batchOfQnsAsWordIDs, qnLengths, [img_vec]
        
    def _mapQnToIDs(self, qn):
        #Convert str question to a list of word_ids
        idList = []
        for word in word_tokenize(qn):
            word = word.strip().lower()
            if self.config.removeQnMark and word is '?':
                continue
            if word in self.mapWordToID:
                    idList.append(self.mapWordToID[word]) 
            else:
                idList.append(self.mapWordToID[self.config.unkWord])
        return idList
    
    def destruct(self):
        self.imgData.close()
