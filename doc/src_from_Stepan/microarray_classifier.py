import pandas as pd
import numpy as np

from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.preprocessing import *

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

import matplotlib.pyplot as plt


# Load TCGA RNA-seq data
df = pd.read_csv("data/BRCA_U133A.csv", index_col=0)
print(df)
# Take 8-gene signature
genes = ["CDK1", "FOXM1", "LRIG2", "MSH2", "PLK1", "RACGAP1", "RRM2", "TMPO"]
genes = ["BUB1B", "KIF4A", "PPFIA1"]

X_train = df.loc[df["Dataset"].isin(["GSE3494", "GSE6532"]), genes].to_numpy()
y_train = df.loc[df["Dataset"].isin(["GSE3494", "GSE6532"]), "Class"].to_numpy()

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# Linear SVM
classifier = SVC(kernel="linear", class_weight="balanced", probability=True)

# Cross-validation
parameters_to_optimize = {"C": np.logspace(-4, 4, 9, base=4)}
splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=100)

CV = GridSearchCV(
    classifier,
    parameters_to_optimize,
    scoring="balanced_accuracy",
    cv=splitter,
    refit=True,
    iid=False)
CV.fit(X_train, y_train)


def TPR(y_true, y_pred):
    M = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return TP / (TP + FN)


def TNR(y_true, y_pred):
    M = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return TN / (TN + FP)


for dataset in ["GSE12093", "GSE17705", "GSE1456"]:
    X_test = df.loc[df["Dataset"] == dataset, genes].to_numpy()
    y_test = df.loc[df["Dataset"] == dataset, "Class"].to_numpy()
    X_test = scaler.transform(X_test)

    y_pred = CV.predict(X_test)
    y_proba = CV.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    print("Dataset: {}".format(dataset))
    print("ROC AUC: {}".format(roc_auc))
    print("Sensitivity: {}".format(TPR(y_test, y_pred)))
    print("Specificity: {}".format(TNR(y_test, y_pred)))
    print("*"*17)

    plt.title('ROC curve, {}'.format(dataset))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specificity')
    plt.savefig("ROC_{}.pdf".format(dataset))
    plt.close()
