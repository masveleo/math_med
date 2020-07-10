import pandas as pd
import numpy as np

from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.preprocessing import *

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

# Load TCGA RNA-seq data
df = pd.read_csv("data/TCGA.tsv", sep="\t", index_col=0)
print(df)

# Take 8-gene signature
genes = ["CDK1", "FOXM1", "LRIG2", "MSH2", "PLK1", "RACGAP1", "RRM2", "TMPO"]
X = df[genes].to_numpy()
y = df["Class"].to_numpy()

# Establish a classification pipeline
scaler = StandardScaler()
classifier = Pipeline([
    ("scaler", scaler),
    ("SVM", SVC(kernel="linear", class_weight="balanced"))
])

# Cross-validation
parameters_to_optimize = {"SVM__C": np.logspace(-4, 4, 9, base=4)}

def TPR(y_true, y_pred):
    M = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return TP / (TP + FN)

def TNR(y_true, y_pred):
    M = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return TN / (TN + FP)


scoring = {
    "AUC": "roc_auc",
    "Balanced Accuracy": "balanced_accuracy",
    "Sensivity": make_scorer(TPR),
    "Specificity": make_scorer(TNR),
}

splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20)

CV = GridSearchCV(
    classifier,
    parameters_to_optimize,
    scoring=scoring,
    cv=splitter,
    refit=False,
    iid=False)
CV.fit(X, y)

# Infer best parameter
mean_test_scoring_values = {s: CV.cv_results_["mean_test_" + s] for s in scoring}
max_BA_index = np.argmax(mean_test_scoring_values["Balanced Accuracy"])
best_scores = {s: mean_test_scoring_values[s][max_BA_index] for s in scoring}

print({s: mean_test_scoring_values[s][max_BA_index] for s in scoring})
print(CV.cv_results_["params"][max_BA_index])
