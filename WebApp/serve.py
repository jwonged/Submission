'''
Created on 21 Mar 2018

@author: jwong
'''

from model.Image_AttModel import ImageAttentionModel
from model.Attention_LapConfig import Attention_LapConfig
from model.InputProcessor import InputProcessor, TestProcessor
from model.model_utils import OutputGenerator
from model.InputProcessor import OnlineProcessor
from model.ImageProcessor import ImageProcessor

def get_model_api():
    
    config = Attention_LapConfig(load=True)
    model = ImageAttentionModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath) 
    processor = OnlineProcessor(config.trainImgFile, config)
    outputGenerator = OutputGenerator('/media/jwong/Transcend/VQADataset/TrainSet/trainImgPaths.txt')
    
    def model_api(input_qn, img_path, usePreprocessedImg=False, img_id = 214587):
        if usePreprocessedImg:
            #/media/jwong/Transcend/VQADataset/TrainSet/Images_train2014/COCO_train2014_000000214587.jpg
            #call model predict function
            alpha, pred = model.solve(input_qn, str(img_id), processor)
            alp_imgName = outputGenerator.getSingleOutput(alpha, img_id, input_qn, pred)
            
            #/media/jwong/Transcend/VQADataset/TrainSet/Images_train2014/COCO_train2014_000000299333.jpg
            output_data = {"input": input_qn,
                           "ans": pred, 
                           "alpha": alp_imgName}
            return output_data
        else:
            #call model predict function
            alpha, pred = model.solve(input_qn, img_path, processor)
            print('Prediction made: {}'.format(pred))
            alp_imgName = outputGenerator.getSingleOutput(alpha, img_path, input_qn, pred)
            print('Palpha made: {}'.format(alp_imgName))
            output_data = {"input": 'Question: ' + input_qn,
                           "ans": 'Prediction: ' + pred, 
                           "alpha": alp_imgName}
            return output_data
    
    #return lambda func
    return model_api