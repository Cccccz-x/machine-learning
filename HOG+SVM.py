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
train_image_path = 'train'
test_image_path = 'test'
train_label_path = os.path.join('train','train.txt') #train\train.txt
test_label_path = os.path.join('test','test.txt')

image_height = 128
image_width = 100

train_feat_path = 'trainfeature/' #文件夹
test_feat_path = 'testfeature/'
model_path = 'model/'

def get_image_list(filePath, nameList):
    print('read image from', filePath)
    img_list = []
    for name in nameList:
        temp = Image.open(os.path.join(filePath, name + '.jpg'))
        img_list.append(temp.copy())
        temp.close()
    return img_list

def get_feat(image_list, name_list, label_list, savePath):
    i = 0
    for image in image_list:
        try:
            image = np.reshape(image,(image_height, image_width, 3)) #每个图像是三维矩阵，第三维大小为3，即RGB
        except:
            print('the size of image error:',name_list[i])
            continue
        gray = rgb2gray(image)/255.0

        fd = hog(gray, orientations=12, block_norm='L1', pixels_per_cell=[8,8], cells_per_block=[4,4], visualize=False, transform_sqrt=True)
        fd = np.concatenate((fd, [label_list[i]])) #拼接
        fd_name = name_list[i] + '.feat'
        fd_path = os.path.join(savePath, fd_name)
        joblib.dump(fd, fd_path) #保存
        i += 1
    print("test feature are extracted and saved")


def rgb2gray(im):
    gray = im[:,:,0]*0.2989+im[:,:,1]*0.5870+im[:,:,2]*0.1140
    return gray

def get_name_label(file_path):
    print("read label from",file_path)
    name_list = []
    label_list = []
    with open(file_path) as f:
        for line in f.readlines():
            if len(line) >= 3: #读出line是字符串
                name_list.append(line.split(' ')[0])
                label_list.append(line.split(' ')[1].replace('\n','').replace('\r',''))
                if not str(label_list[-1].isdigit()):
                    print("label must be num, but:",label_list[-1],"error and stop")
                    exit(1)
    return name_list, label_list

def extra_feat():
    train_name, train_label = get_name_label(train_label_path)
    test_name, test_label = get_name_label(test_label_path)

    train_image = get_image_list(train_image_path, train_name)
    test_image = get_image_list(test_image_path, test_name)
    get_feat(train_image, train_name, train_label, train_feat_path)
    get_feat(test_image, test_name, test_label, test_feat_path)

def mkdir():
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)

def train_and_test():
    t0 = time.time()
    features = []
    labels = []
    correct_number = 0
    total = 0
    for feat_path in glob.glob(os.path.join(train_feat_path,'*.feat')): #用glob来匹配
        data = joblib.load(feat_path)
        features.append(data[:-1])
        labels.append(data[-1])
    print("Training a Linear SVM Classifier")
    clf = LinearSVC()
    clf.fit(features, labels)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    joblib.dump(clf, model_path + 'model')

    #clf = joblib.load(model_path + 'model')

    result_list = []
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        total += 1
        if platform.system() == 'Windows':
            symbol = '\\' #转义
        else:
            symbol = '/'
        image_name = feat_path.split(symbol)[1].split('.feat')[0]
        data_test = joblib.load(feat_path)
        data_test_feat = data_test[:-1].reshape((1,-1)).astype(np.float64)
        result = clf.predict(data_test_feat)
        result_list.append(image_name + ' ' + label_map[int(result[0])]+ '\n')
        if int(result[0]) == int(data_test[-1]):
            correct_number += 1
    write_to_txt(result_list)
    rate = float(correct_number) / total
    t1 = time.time()
    print('accuracy:%f' %rate)
    print('time used:%f' %(t1-t0))

def write_to_txt(list):
    with open('result.txt','w') as f:
        f.writelines(list)
    print('result store in result.txt')

if __name__ == '__main__':
    mkdir()
    extra_feat()
    train_and_test()










































