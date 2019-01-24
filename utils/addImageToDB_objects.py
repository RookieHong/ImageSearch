# Add a image to Database so it can be searched(based on objects)
import importlib
import pickle

def addImageToDB(imgPath, selectedPredictor):
    DBdir = '../Data/objects-features-{}/'.format(selectedPredictor)
    predictor = importlib.import_module('predictors.{}'.format(selectedPredictor))

    predictions, features = predictor.predictionAndFeature(imgPath)
    for i, [x1, y1, x2, y2, label, conf] in enumerate(predictions):
        imgDict = {'imgPath': imgPath,
                   'feature': features[i],
                   'x1': x1,
                   'y1': y1,
                   'x2': x2,
                   'y2': y2,
                   'conf': conf}

        featureFile = open(DBdir + label, 'ab+')
        pickle.dump(imgDict, featureFile)
        featureFile.close()

    return predictions