import task1
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score,precision_score,f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
# f1 score will lead zero-division warning because some class may not be predicted
warnings.filterwarnings("ignore")

def main():
    (data_list,test_data_list,keep_num_list),label,test_label = task1.main()
    keep_list = [0.4,0.6,0.8,1.]
    # print(len(data_list),len(test_data_list),len(keep_list),len(keep_num_list))
    
    linear_p = []
    linear_r = []
    linear_f1 = []
    linear_acc = []
    for i,data in enumerate(data_list):
        model = SVC(kernel='linear')
        scores = cross_val_score(model, data, label, cv=5,scoring='accuracy')
        print(f'linear kernel SVM + {keep_list[i]*100}% data feature({keep_num_list[i]} dimension) 5-fold cross validation scores: {scores}')
        model = SVC(kernel='linear')
        model.fit(data,label)
        acu_test = model.score(test_data_list[i], test_label)
        y_pred = model.predict(test_data_list[i])
        precision = precision_score(test_label, y_pred, average="macro")
        recall = recall_score(test_label, y_pred, average="macro")
        f1 = f1_score(test_label, y_pred, average="macro")
        linear_p.append(precision)
        linear_r.append(recall)
        linear_f1.append(f1)
        linear_acc.append(acu_test)
    
    plt.figure(figsize=(12,10))
    plt.plot(np.log(keep_num_list),linear_p,label='precision',marker='*')
    plt.plot(np.log(keep_num_list),linear_r,label='recall',marker='o')
    plt.plot(np.log(keep_num_list),linear_f1,label='f1 score',marker='2')
    plt.plot(np.log(keep_num_list),linear_acc,label='accuracy',marker='s')
    plt.xlabel('log(feature vector dim)',fontsize=15)
    plt.xticks(np.log(keep_num_list),[f'log({keep_num})' for keep_num in keep_num_list])
    plt.legend(loc='best',fontsize=12)
    plt.savefig('svm_plot.png')

    C_list = [0.2,0.5,1.0,2.0,5.0,10.0]
    data = data_list[-2]
    test_data = test_data_list[-2]
    rbf_p,rbf_r,rbf_f1,rbf_acc = [],[],[],[]
    for c in C_list:
        model = SVC(kernel='rbf',C=c)
        model.fit(data,label)
        acu_test = model.score(test_data, test_label)
        y_pred = model.predict(test_data)
        precision = precision_score(test_label, y_pred, average="macro")
        recall = recall_score(test_label, y_pred, average="macro")
        f1 = f1_score(test_label, y_pred, average="macro")
        rbf_p.append(precision)
        rbf_r.append(recall)
        rbf_f1.append(f1)
        rbf_acc.append(acu_test)
    
    plt.figure(figsize=(12,10))
    plt.plot(C_list,rbf_p,label='precision',marker='*')
    plt.plot(C_list,rbf_r,label='recall',marker='o')
    plt.plot(C_list,rbf_f1,label='f1 score',marker='2')
    plt.plot(C_list,rbf_acc,label='accuracy',marker='s')
    plt.xticks(C_list,C_list)
    plt.xlabel('regularization hyper-param',fontsize=15)
    plt.legend(loc='best',fontsize=12)
    plt.savefig('rbf_svm_plot.png')



if __name__ == "__main__":
    main()