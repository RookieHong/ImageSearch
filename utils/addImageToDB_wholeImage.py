# Add a image to Database so it can be searched(based on whole image search)
import pickle
import importlib

def addImageToDB(imgPath, selectedPredictor, DBdir):
    predictor = importlib.import_module('predictors.{}'.format(selectedPredictor))

    prob, label, feature = predictor.predictionAndFeature(imgPath)
    imgDict = {'imgPath': imgPath, 'feature': feature}

    featureFile = open(DBdir + label, 'ab+')
    pickle.dump(imgDict, featureFile)
    featureFile.close()

    return label