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
    return np.argmax(aucLst), aucLst

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

''' gets a rough idea where the best parameters lie '''
boosting_grid_rough = {'learning_rate': np.logspace(-3, 0, num = 4),
                        'max_depth': [2, 3, 5],
                        'n_estimators': [10, 30, 100, 300, 1000]}

df=search(x,y,boosting_grid_rough,3)
df.sort_values(by='mean_test_score').tail(10)

# results will vary, but top 10 results on this machine:
#'learning_rate': 0.01 - .1
#'max_depth': [2, 3, 5],
#'n_estimators': [100, 300, 1000]

boosting_grid_med = {'learning_rate': np.logspace(-2.5, -.5, num = 5),
                        'max_depth': [1, 4, 7, 10],
                        'n_estimators': [100, 300, 1000, 3000, 6000]}

df=search(x,y,boosting_grid_med,3)
df.sort_values(by='mean_test_score').tail(10)

# results will vary, but top 20 results on this machine:
#'learning_rate': 0.00316228 - 0.316228
#'max_depth': [1, 4, 7, 10],
#'n_estimators': [100, 300, 1000]

boosting_grid_med2 = {'learning_rate': np.logspace(-3, -1, num = 5),
                        'max_depth': [4, 5, 6, 7],
                        'n_estimators': [300, 1000, 3000, 4500, 6800, 10000]}

df=search(x,y,boosting_grid_med2,3)
df.sort_values(by='mean_test_score').tail(10)

# results will vary, but top 20 results on this machine:
#'learning_rate': 0.001 - 0.0316228
#'max_depth': [4,5,6],
#'n_estimators': [300, 1000, 3000, 4500, 6800, 10000]

####################
#final tuning

#it seems likely that the optimal occurs at >16000 trees; this is probably infeasible to run
#due to variance, the max in this dataset occurs at 15998 trees

kf = KFold(n_splits=10, shuffle=True, random_state=646719267)
aucs2=np.zeros([10,16000])
row=0
for trainIdx, testIdx in kf.split(range(len(X))):
    gb = GradientBoostingClassifier(**{'learning_rate': 0.003, 'max_depth': 5, 'n_estimators': 16000})
    gb.fit(X.iloc[trainIdx], y.iloc[trainIdx])
    aucLst=[]
    for preds in gb.staged_predict_proba(X.iloc[testIdx]):
        aucLst.append(roc_auc_score(y.iloc[testIdx], preds[:,1]))
    aucs2[row]=aucLst
    row+=1
aucLst2 = aucs2.mean(axis=0)
plt.scatter(range(len(aucLst2)), aucLst2, marker='.')
plt.scatter(np.argmax(aucLst2), max(aucLst2), marker='o', c='red')
plt.xlabel('number of trees')
plt.ylabel('mean AUC')
plt.title('max depth = 5, learning rate = 0.003')
plt.show()
#max occurs at around 5757 trees
#mean AUC at this point is around 0.9838914306225004

### final validation
gb = GradientBoostingClassifier(**{'learning_rate': 0.003, 'max_depth': 5, 'n_estimators': 5757})
gb.fit(X, y)
Xtest = pd.read_json('data/testFeats.json')
ytest = pd.read_json('data/testLabel.json', typ='Series', dtype=False)
preds = gb.predict_proba(Xtest)
auc = roc_auc_score(ytest, preds[:,1])
#0.9859929921359716

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