'''
Created on 15 Jan 2018

@author: jwong
'''
import numpy as np
import re
import json
from collections import Counter

def getPretrainedw2v(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with np.load(filename) as data:
        return data["vectors"]

def resolveAnswer(possibleAnswersList):
    #answers = []
    #for answerDetails in possibleAnswersList:
    #    answers.append(answerDetails['answer'])
    mostCommon = Counter(possibleAnswersList).most_common(1)
    return mostCommon[0][0]

def generateForSubmission(qn_ids, preds, jsonFile):
        '''
        result{
            "question_id": int,
            "answer": str
        }'''
        results = []
        for qn_id, pred in zip(qn_ids, preds):
            singleResult = {}
            singleResult["question_id"] = int(qn_id)
            singleResult["answer"] = str(pred)
            results.append(singleResult)
        
        with open(jsonFile, 'w') as jsonOut:
            print('Writing to {}'.format(jsonFile))
            json.dump(results, jsonOut)

import matplotlib.pyplot as plt
import skimage.transform
import cv2
from scipy import ndimage
import numpy as np
import random
class OutputGenerator(object):
    def __init__(self, imgPathsFile):
        self.idToImgpathMap = {}
        #print('Reading {}'.format(imgPathsFile))
        #with open(imgPathsFile, 'r') as reader:
        #    for image_path in reader:
        #        image_path = image_path.strip()
        #        self.idToImgpathMap[str(self.getImageID(image_path))] = image_path
    
    def getImageID(self,image_path):
        #Get image ID
        splitPath = image_path.split('/')
        imgNameParts = splitPath[len(splitPath)-1].split('_') #COCO, train, XXX.jpg
        suffix =  imgNameParts[len(imgNameParts)-1] #XXX.jpg
        img_id = int(suffix.split('.')[0])
        return img_id
    
    def convertIDtoPath(self, img_id):
        return self.idToImgpathMap[img_id]
    
    def displaySingleOutput(self, alpha, img_id, qn, pred):
        print('image path: {}'.format(img_id))
        imgvec = ndimage.imread(img_id)
        imgvec = cv2.resize(imgvec, dsize=(448,448))
        
        alp_img = skimage.transform.pyramid_expand(
            alpha.reshape(14,14), upscale=32, sigma=20)
        alp_img = np.transpose(alp_img, (1,0))
        
        plt.subplot(1,1,1)
        plt.title('Qn: {} pred: {}'.format(qn, pred))
        plt.imshow(imgvec)
        plt.imshow(alp_img, alpha=0.80)
        plt.axis('off')
        
        #plt.subplot(2,1,1)
        #plt.title('Qn: {}, pred: {}'.format(qn, pred))
        #plt.imshow(imgvec)
        #plt.axis('off')
            
        plt.show()
    
    def getSingleOutput(self, alpha, img_id, qn, pred):
        print('Num of images: {}'.format(img_id))
        imgvec = ndimage.imread(img_id)
        imgvec = cv2.resize(imgvec, dsize=(448,448))
        
        alp_img = skimage.transform.pyramid_expand(
            alpha.reshape(14,14), upscale=32, sigma=40)
        alp_img = np.transpose(alp_img, (1,0))
        
        plt.subplot(1,1,1)
        plt.title('Qn: {} pred: {}'.format(qn, pred))
        plt.imshow(imgvec)
        plt.imshow(alp_img, alpha=0.80)#, cmap='gray')
        plt.axis('off')
        name = 'yolo{}.png'.format(random.randint(1,100))
        plt.savefig(name)
        return name
        
    def displayOutput(self, alphas, img_ids, qns, preds):
        
        
        print('Num of images: {}'.format(img_ids))
        for n, (alp, img_id, qn, pred) in enumerate(zip(alphas, img_ids, qns, preds)):
            if n>2:
                break
            imgvec = ndimage.imread(self.idToImgpathMap[img_id])
            imgvec = cv2.resize(imgvec, dsize=(448,448))
            
            alp_img = skimage.transform.pyramid_expand(
                alp.reshape(14,14), upscale=32, sigma=20)
            alp_img = np.transpose(alp_img, (1,0))
            
            print(n)
            plt.subplot(2,4,(n+1))
            plt.title('Qn: {}, pred: {}'.format(qn, pred))
            plt.imshow(imgvec)
            plt.imshow(alp_img, alpha=0.80)
            plt.axis('off')
            
            plt.subplot(2,4,(n+1)*2)
            plt.title('Qn: {}, pred: {}'.format(qn, pred))
            plt.imshow(imgvec)
            plt.axis('off')
            
        plt.show()
