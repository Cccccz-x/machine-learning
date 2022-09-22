import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC

train_path="diabetes_train.data"
test_path = "diabetes_test.data"
train_data = np.array(pd.read_csv(train_path, header=None, sep=','))
test_data = np.array(pd.read_csv(test_path,header=None, sep=','))
print(train_data)

train_data[train_data == 'tested_positive'] = 1
train_data[train_data == 'tested_negative'] = -1
test_data[test_data == 'tested_positive'] = 1
test_data[test_data == 'tested_negative'] = -1
train_label = train_data[:,-1].astype(np.int64)
# test_label = test_data[:,-1].astype(np.int64)

correct_number = 0
clf = LinearSVC(max_iter=100000)
clf.fit(train_data[:,:-1], train_label)

result = clf.predict(test_data[:,:-1])
for i in range(len(test_data)):
    print(result[i],test_data[i,-1])
    if int(result[i]) == int(test_data[i,-1]):
        correct_number += 1
rate = float(correct_number) / len(test_data)
print('accuracy:%f' %rate)

