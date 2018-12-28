import pickle
import os

# featureFilesPath = '../Data/ImageNet/ILSVRC2012/val-features/'
# featureFiles = os.listdir(featureFilesPath)
# for featureFile in featureFiles:
#     f = open(featureFilesPath + featureFile, 'rb')
#     data = pickle.load(f)
#     print(data)

a={'path':'/asd/asdasd/', 'val': 123456}
b={'path':'/qwe/qweqwe/', 'val': 456789}

f=open('test', 'ab+')
pickle.dump(a,f)
f.close()

f=open('test','ab+')
pickle.dump(b,f)
f.close()

f=open('test','rb')
try:
    while True:
        data = pickle.load(f)
        print(data)
except EOFError as e:
    pass