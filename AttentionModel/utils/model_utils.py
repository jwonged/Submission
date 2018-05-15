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
    answers = []
    for answerDetails in possibleAnswersList:
        answers.append(answerDetails['answer'])
    mostCommon = Counter(answers).most_common(1)
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
