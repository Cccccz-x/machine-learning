import glob
import platform
import time
from PIL import Image
from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC

label_map = {0:'cat',
             1:'dog'}

train_path = 'train'
test_path = 'test'
train_label_path = os.path.join(train_path,'train.txt')
test_label_path = os.path.join(test_path,'test.txt')

img_height = 128
img_width = 100

train_feat_path = 'trainfeature/' #文件夹
test_feat_path = 'testfeature/'
model_path = 'model/'

def get_name_label(path):
    name = []
    label = []
    with open(path) as f:
        for i in f.readlines():
            if len(i)>=3:
                name.append(i.split(' ')[0])
                label.append(i.split(' ')[1])
            else:
                print("len i should >= 3 , error")

    return name, label

def get_image(path, name_list):
    img_list = []
    for name in name_list:
        image = Image.open(os.path.join(path, name + '.jpg'))
        img_list.append(image.copy())
        image.close()
    return img_list

def get_feat(img_list,name_list, label_list, path):
    i = 0
    for img in img_list:
        try:
            img = np.reshape(img,(img_height,img_width,3))
        except:
            print(name_list[i],"type error")

        gray = rgb2gray(img)/255.0
        feat = hog(gray, orientations=12, block_norm='L1', pixels_per_cell=[4,4], cells_per_block=[2,2], visualize=False, transform_sqrt=True)
        feat_name = name_list[i] + '.feat'
        feat = np.concatenate((feat, [label_list[i]]))
        feat_path = os.path.join(path, feat_name)
        joblib.dump(feat, feat_path)
        i+=1


def rgb2gray(img):
    return img[:,:,0]*0.2989+img[:,:,1]*0.5870+img[:,:,2]*0.1140

def extra_feat():
    train_name, train_label = get_name_label(train_label_path)
    test_name, test_label = get_name_label(test_label_path)
    train_image = get_image(train_path, train_name)
    test_image = get_image(test_path, test_name)
    get_feat(train_image, train_name, train_label, train_feat_path)
    get_feat(test_image, test_name, test_label, test_feat_path)

def mkdir():
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)

def train_and_test():
    t0 = time.time()
    total = 0
    correct_num = 0
    train_feat = []
    train_label = []
    for feat in glob.glob(os.path.join(train_feat_path, '*.feat')):
        train_feat.append(joblib.load(feat)[:-1])
        train_label.append(joblib.load(feat)[-1])

    clf = LinearSVC()
    clf.fit(train_feat, train_label)

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    joblib.dump(clf,model_path + 'model')

    result_list = []
    for feat_path in glob.glob(os.path.join(test_feat_path,'*.feat')):
        total += 1
        if platform.system() == 'Windows':
            symbol = '\\'
        else:
            symbol = '/'
        image_name = feat_path.split(symbol)[1].split('.')[0]
        test_feat = joblib.load(feat_path)[:-1].reshape((1,-1)).astype(np.float64)
        test_label = joblib.load(feat_path)[-1]
        result = clf.predict(test_feat)
        result_list.append(image_name + ' ' + label_map[int(result[0])] + '\n')

        if result[0] == test_label:
            correct_num += 1

    write_to_txt(result_list)
    rate = float(correct_num)/total
    t1 = time.time()
    print("accuracy:",rate)
    print("used time:",t1-t0)

def write_to_txt(list):
    with open('result.txt','w') as f:
        f.writelines(list)
    print("result store in result.txt")

if __name__ == '__main__':
    mkdir()
    extra_feat()
    train_and_test()
