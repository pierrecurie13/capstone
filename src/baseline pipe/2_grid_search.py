import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.externals.joblib import parallel_backend
import matplotlib.pyplot as plt

dirName = 'data/training2017/'
xTrain = np.load(dirName + 'trainFeat.npy')
yTrain = np.load(dirName + 'trainlabel.npy')
yTrain=yTrain[:,1]
xVal = np.load(dirName + 'validFeat.npy')
yVal = np.load(dirName + 'validlabel.npy')
yVal=yVal[:,1]

def score_func(y, y_pred, **kwargs):
    mask = y!='~'
    y=y[mask]
    y_pred=y_pred[mask]
    y_pred=y_pred[:,0]/(y_pred[:,0]+y_pred[:,1])
    mask = np.isnan(y_pred)
    y_pred[mask]=.5
    mask = y=='A'
    y = np.zeros(len(y))
    y[mask]=1
    return roc_auc_score(y, y_pred)

scorer = make_scorer(score_func, needs_proba=True)

# def findNum(x,y,params, cv=3, seed=646719267):
#     '''
#     params is dict
#     '''
#     kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
#     aucs=np.zeros([cv,params['n_estimators']])
#     row=0
#     for trainIdx, testIdx in kf.split(range(len(x))):
#         gb = GradientBoostingClassifier(**params)
#         gb.fit(x[trainIdx], y[trainIdx])
#         aucLst=[]
#         for preds in gb.staged_predict_proba(x[testIdx]):
#             aucLst.append(score_func(y[testIdx], preds))
#         aucs[row]=aucLst
#         row+=1
#     aucLst = aucs.mean(axis=0)
#     return np.argmax(aucLst), aucLst

def findNum2(xTrain,yTrain,xTest,yTest,params={}):
    '''
    params is dict
    '''
    gb = GradientBoostingClassifier(**params)
    gb.fit(xTrain, yTrain)
    aucLst=[]
    for preds in gb.staged_predict_proba(xTest):
        aucLst.append(score_func(yTest, preds))
    return aucLst

def plotFindNum(aucLst, params='title'):
    plt.scatter(range(len(aucLst)), aucLst, marker='.')
    plt.scatter(np.argmax(aucLst), max(aucLst), marker='o', c='red')
    plt.xlabel('number of trees')
    plt.ylabel('mean AUC')
    plt.title(params)
    plt.show()
    
# def search(x,y,params,cv=5):

#     clf = GridSearchCV(GradientBoostingClassifier(), params, scorer,
#                                 n_jobs=3, cv=cv, verbose=1)
#     print("Starting grid search - coarse (will take several minutes)")
#     with parallel_backend('threading'):
#         clf.fit(x,y)
#     return pd.DataFrame(clf.cv_results_)

def search(xTrain,yTrain,xTest,yTest,lrnRates,params={}):
    out=[]
    for lrnRate in lrnRates:
        params.update({'learning_rate': lrnRate})
        aucLst = findNum2(xTrain,yTrain,xTest,yTest,params)
        num=np.argmax(aucLst)
        out.append([lrnRate, params['max_depth'], num+1, aucLst[num]])
    return out

df=pd.DataFrame(search(xTrain,yTrain,xVal,yVal,np.logspace(-3,0,num=4),
    params={'max_depth':2,'n_estimators':1000}),
    columns=['learning_rate','max_depth','n_estimators','validation AUC'])
df.sort_values(by='validation AUC')

#   learning_rate  max_depth  n_estimators  validation AUC
#          0.001          4           956        0.973071
#          1.000          4             3        0.973416
#          0.010          4           375        0.977147
#          0.100          4            45        0.977802
# similar results for max depth of 2,3,5,6
# too similar - leak?

### final validation
gb = GradientBoostingClassifier(**{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 45})
gb.fit(np.vstack([xTrain,xVal]), np.hstack([yTrain,yVal]))
xTest = np.load(dirName + 'testFeat.npy')
yTest = np.load(dirName + 'testlabel.npy')[:,1]
preds = gb.predict_proba(xTest)
auc = score_func(yTest, preds)
#0.9783068783068782

mask = gb.predict(xTest)==yTest


fpr, tpr, thresholds = roc_curve(ytest, preds[:,1], pos_label=1)
plt.scatter(fpr,tpr, marker='.')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()

###
#train final model
X2 = X.append(Xtest)
y2 = y.append(ytest)
gb2 = GradientBoostingClassifier(**{'learning_rate': 0.003, 'max_depth': 5, 'n_estimators': 5757})
gb2.fit(X2, y2)
with open('data/model_final.p','wb') as f:
    pickle.dump(gb2, f)