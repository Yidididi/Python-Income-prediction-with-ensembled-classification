# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 18:17:12 2019

@author: Yidi Kang
"""

##########################################
# Praperation
import pandas as pd
import numpy as np
import math
#from collections import Counter
#from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

##Different resampling methods 
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

##Different classifier methods
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

##Different demision reduction methods
# from sklearn.neural_network import BernoulliRBM 
# from sklearn.decomposition import KernelPCA
# from sklearn.decomposition import PCA
# from sklearn.decomposition import SparsePCA
# from sklearn.decomposition import NMF

##########################################
#  load the train and test data
train_dat = pd.read_csv("census-income.data.csv",header = None, na_values=" ?")
test_dat = pd.read_csv("census-income.test",header = None, na_values=" ?")
#read the variable information
with open("census-income.names","r") as names:
    i = 1
    line = names.readline().replace('\n', '') 
    while i <= 112:
        print(line)
        line = names.readline().replace('\n', '') 
        i += 1
#rename the columns
names = ["age","workclass","fnlwgt","education",
         "education_num","marital-status","occupation","relationship",
         "race","sex","capital-gain","capital-loss","hours-per-week",
         "native-country","label"]
train_dat.columns = names
test_dat.columns = names
#make train and test data match 
test_dat['label'] = [i.replace('.','') for i in test_dat['label']]

##########################################
# EDA + Data Pre-processing
#get the basic information of the raw data
train_dat.head()
test_dat.head()
train_dat.describe()
test_dat.describe()

#NA Checking
train_dat.isnull().sum()
test_dat.isnull().sum()
train_dat.fillna(' other',inplace = True)
test_dat.fillna(' other',inplace = True)

# countries to "US" and "other"
train_dat["native-country"] = [' other' if i!=' United-States' else i for i in train_dat["native-country"]]
test_dat["native-country"]= [' other' if i!=' United-States' else i for i in test_dat["native-country"]]

##########################################
# Feature engineering
# delete education 
train_dat[["education_num",'education']]
del train_dat['education']
del test_dat['education']

# Get dummies
train = pd.get_dummies(train_dat, prefix_sep='_', drop_first=True)
# train head
train.head()
# Get dummies
test = pd.get_dummies(test_dat, prefix_sep='_', drop_first=True)

# Normalization steps
con_list = ["age","fnlwgt","education_num","capital-gain","capital-loss","hours-per-week"]
scaler = StandardScaler()  #Normalize data (zero mean, unit variance data)
train_norm = scaler.fit_transform(train_dat[con_list])  #Fit to data, then transform it.
train_norm = pd.DataFrame(train_norm,columns=con_list) 
train_dat[con_list]=train_norm
test_norm = scaler.fit_transform(test_dat[con_list])  #Fit to data, then transform it.
test_norm = pd.DataFrame(test_norm,columns=con_list) 
test_dat[con_list]=test_norm

# check column different
set(test.columns)-set(train.columns)
set(train.columns)-set(test.columns)
#estract label and data
train_label = train['label_ >50K']
del train['label_ >50K']
test_label = test['label_ >50K']
del test['label_ >50K']

##########################################
# Methodology
# classifiers building
KNN = KNeighborsClassifier(n_neighbors=3)
SVM = svm.SVC(gamma='scale')
GB = GradientBoostingClassifier()
ET = ExtraTreesClassifier(n_estimators=100)
DNN = MLPClassifier(hidden_layer_sizes = (100,3),solver = "adam")
RF = RandomForestClassifier()
LG = linear_model.LogisticRegression(C=10000.0,solver = "lbfgs")

classifier_list = [KNN,SVM,GB,ET,DNN,RF,LG]
classifier_name = ["KNN","SVM","GradientBoost","ExtraTrees","DNN","RandomForest","LogisticRegression"]

'''
##dimesion reduction methods
rbm = BernoulliRBM(n_components=100, learning_rate= 0.01, n_iter = 30, verbose = True)
kpca = KernelPCA(n_components=3, random_state=0)
pca = PCA(n_components=3, random_state=0)
spca = SparsePCA(n_components=3, random_state=0,normalize_components=True)
nmf = NMF(n_components=3,random_state=0)
dimReduct_list = [rbm,kpca,pca,spca,nmf]
dimReduct_name = ["RNM","KPCA","PCA","SPCA","NMF"]
# We decided not to do feature demension reducing. 
'''

#resampling methods 
cc = ClusterCentroids(random_state=42)
smote_enn = SMOTEENN(random_state=0)
ad = ADASYN(random_state = 0)
ros = RandomOverSampler(random_state = 0)
smote = SMOTE(random_state = 0)

resampling_list = [cc,smote_enn,ad,ros,smote]
resampling_name = ["ClusterCentroids","SMOTEENN","ADASYN","RandomOverSampler","SMOTE"]

##########################################
# Model evaluation
# define the measure function
def compute_measure(predicted_label,true_label):
    t_idx = (predicted_label==true_label) #true predicted
    # f_idx = np.logical_not(t_idx) #false predicted
    
    p_idx = (true_label > 0) #postive targets 
    n_idx = np.logical_not(p_idx) #negative targets
    
    tp = np.sum(np.logical_and(t_idx,p_idx)) #TP
    tn = np.sum(np.logical_and(t_idx,n_idx)) #TN
    
    #false positive: original negative but classified as positive
    fp = np.sum(n_idx) - tn
    
    #false negative: original positive but classified as negative
    fn = np.sum(p_idx) - tp
    
    tp_fp_tn_fn_list = []
    tp_fp_tn_fn_list.append(tp)
    tp_fp_tn_fn_list.append(fp)
    tp_fp_tn_fn_list.append(tn)
    tp_fp_tn_fn_list.append(fn)
    tp_fp_tn_fn_list = np.array(tp_fp_tn_fn_list)  
    tp = tp_fp_tn_fn_list[0]
    fp = tp_fp_tn_fn_list[1]
    tn = tp_fp_tn_fn_list[2]
    fn = tp_fp_tn_fn_list[3]
    
    #sensitivity the percentage of positive subjects correctly predicted
    with np.errstate(divide = 'ignore'):
        sen = (1.0*tp)/(tp+fn)
    
    #specificity the percentage of negative subjects correctly predicted
    with np.errstate(divide = 'ignore'):
        spc = (1.0*tn)/(tn+fp)
    
    #PPR positive subjects predictive ratio
    with np.errstate(divide = 'ignore',invalid = 'ignore'):
        ppr = (1.0*tp)/(tp+fp)
        ppr = np.nan_to_num(ppr)
        
    #NPR negative subjects predictive ratio
    with np.errstate(divide = 'ignore',invalid = 'ignore'):
        npr = (1.0*tn)/(tn+fn)
        npr = np.nan_to_num(npr)
    
    #accuracy the percentage of correctly predicted  subjects among all.
    acc = (tp+tn)*1.0/(tp+fp+tn+fn)
    
    #diagnostic index
    d_index = math.log(1+acc,2) + math.log(1+(sen+spc)/2,2)
    
    #recall
    recall = sen
    
    #F-1 score
    with np.errstate(divide = 'ignore'):
        F1 = 2*tp/(2*tp+fp+fn)
    
    ans = pd.DataFrame(np.array([d_index,acc,sen,npr,recall,F1]),
                       index = ['diagnostic index','accuracy','sensitivity',
                                  'NPR','recall','F-1 score'],
                       columns = ['Metric']         ) 
    return ans

##########################################
# Model fitting with cross Validation
def cv_combine(resampling_list,resampling_name,
                   classifier_list,classifier_name,
                   data,label):
    n_resample = len(resampling_list)
    n_classifier = len(classifier_list)
    columns = []
    for j in range(n_classifier):
        columns.append(classifier_name[j])
        for i in range(n_resample):
             columns.append(resampling_name[i]+"_"+classifier_name[j])
    compare = pd.DataFrame(np.zeros([6,n_classifier*(n_resample+1)]),index = ['diagnostic index','accuracy','sensitivity'
                                      ,'NPR','recall','F-1 score'],columns=columns)
    #crossvalidation: 5 Folds
    kf = KFold(n_splits=5,shuffle= True)
    for train, test in kf.split(data):
        train_dat, test_dat, train_label, test_label = data.iloc[train,:], data.iloc[test,:], label[train], label[test]
        for i in range(n_resample):
            print(i)
            train_res,label_res = resampling_list[i].fit_resample(train_dat,train_label)
            for j in range(n_classifier):
                print(j)
                classifier_list[j].fit(train_res,label_res)
                pred = classifier_list[j].predict(test_dat)
                
                compare[resampling_name[i]+"_"+classifier_name[j]] =compare[resampling_name[i]+"_"+classifier_name[j]].values +\
                                                                    compute_measure(pred,test_label)["Metric"].values
        for j in range(n_classifier):
            classifier_list[j].fit(train_dat,train_label)
            pred = classifier_list[j].predict(test_dat)
            compare[classifier_name[j]] = compare[classifier_name[j]].values + \
                                          compute_measure(pred,test_label)["Metric"].values
    return compare/5

com3 = cv_combine(resampling_list,resampling_name,
                   classifier_list,classifier_name,
                   train,train_label)
com3.to_csv("compare3.csv")


com3.iloc[1,:][com3.iloc[1,:]>sorted(com3.iloc[1,:],reverse= True)[7]]
com3.iloc[0,:][com3.iloc[0,:]>sorted(com3.iloc[0,:],reverse= True)[10]]

'''
# without cross validation
def method_combine(resampling_list,resampling_name,
                   classifier_list,classifier_name,
                   train_dat,train_label,
                   test_dat,test_label,
                   resample = False):
    n_resample = len(resampling_list)
    n_classifier = len(classifier_list)
    if resample:
        compare = pd.DataFrame(index = ['diagnostic index','accuracy','sensitivity',
                                  'NPR','recall','F-1 score']) 
        for i in range(n_resample):
            print(i)
            train_res,label_res = resampling_list[i].fit_resample(train_dat,train_label)
            for j in range(n_classifier):
                print(j)
                classifier_list[j].fit(train_res,label_res)
                pred = classifier_list[j].predict(test_dat)
                compare[resampling_name[i]+"_"+classifier_name[j]] = compute_measure(pred,test_label)
        for j in range(n_classifier):
            classifier_list[j].fit(train_dat,train_label)
            pred = classifier_list[j].predict(test_dat)
            compare[classifier_name[j]] = compute_measure(pred,test_label)
    return compare
com = method_combine(resampling_list,resampling_name,
                   classifier_list,classifier_name,
                   train,train_label,
                   test,test_label,
                   resample = True)
com.to_csv("compare2.csv")
com.iloc[0,:]
com.iloc[1,:][com.iloc[1,:]>sorted(com.iloc[1,:],reverse= True)[6]]
com.iloc[0,:][com.iloc[0,:]>sorted(com.iloc[0,:],reverse= True)[6]]
'''


##########################################
# Majority vote steps
resampling_list2 = [smote_enn,smote,smote_enn,smote,ros,smote_enn,smote_enn]
rn = ["SMOTE","SMOTEENN","SMOTE","RandomOverSampler","SMOTEENN"]
classifier_list2 = [SVM,RF,LG,GB,DNN]
cn = ["SVM","RandomForest","LogisticRegression","GradientBoost","DNN"]

for i in range(5):
    print("The",i+1,"model is",cn[i],"with",rn[i])
    
def majorityvote1(resampling_list,rn,
                   classifier_list,cn,
                   train,train_label,
                   test,test_label):
    pred = pd.DataFrame()
    for i in range(len(rn)):
        train_res,label_res = resampling_list[i].fit_resample(train,train_label)
        classifier_list[i].fit(train_res,label_res)
        pred[cn[i]] = classifier_list[i].predict(test)
    final = []
    for i in range(len(test)):
        if sum(pred.loc[i,:]) > 2:
            final.append(1)
        else:
            final.append(0)
    return final

def majorityvote2(classifier_list,cn,
                   train,train_label,
                   test,test_label):
    pred = pd.DataFrame()
    for i in range(len(cn)):
        classifier_list[i].fit(train,train_label)
        pred[cn[i]] = classifier_list[i].predict(test)
    final = []
    for i in range(len(test)):
        if sum(pred.loc[i,:]) > 2:
            final.append(1)
        else:
            final.append(0)
    return final

final1 = majorityvote1(resampling_list2,rn,
                   classifier_list2,cn,
                   train,train_label,
                   test,test_label)
final2 = majorityvote2(classifier_list2,cn,
                   train,train_label,
                   test,test_label)
f1 = compute_measure(final1,test_label)
f1
f2 = compute_measure(final2,test_label)
f2



