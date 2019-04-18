from PIL import Image
import os
import glob

ifResize = False
savePath = '../Data/Oxford-5k/resized_query_images/' if ifResize else '../Data/Oxford-5k/query_images/'
croppedPath = '../Data/Oxford-5k/resized_cropped_query_images/' if ifResize else '../Data/Oxford-5k/cropped_query_images/'

for f in glob.iglob(os.path.join('../Data/Oxford-5k/gt_files', '*_query.txt')):
    query_name, x, y, w, h = open(f).read().strip().split(' ')

    query_name = query_name.replace('oxc1_', '')
    query_name = query_name + '.jpg'

    x = int(float(x))
    y = int(float(y))
    w = int(float(w))
    h = int(float(h))

    img = Image.open('../Data/Oxford-5k/oxbuild_images/' + query_name)
    cropped = img.crop((x, y, min((x + w), img.width), min((y + h), img.height)))
    if ifResize:
        cropped = cropped.resize((224, 224), Image.ANTIALIAS)
        img = img.resize((224, 224), Image.ANTIALIAS)

    save_name = os.path.splitext(os.path.basename(f))[0].replace('_query', '') + '.jpg'
    cropped.save(croppedPath + save_name)
    img.save(savePath + save_name)

