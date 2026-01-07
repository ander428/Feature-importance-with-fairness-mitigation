from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, accuracy_score_difference, false_negative_rate_difference
from fairlearn.metrics import false_positive_rate_difference, true_negative_rate_difference

from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio, accuracy_score_ratio, false_negative_rate_ratio 
from fairlearn.metrics import false_positive_rate_ratio, true_negative_rate_ratio

from sklearn.metrics import confusion_matrix, precision_score, roc_auc_score, brier_score_loss, log_loss
from pycalib.metrics import ECE, classwise_ECE, MCE

from fairlearn.metrics import make_derived_metric

import numpy as np

def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp+fn)

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn+fp)

def prop_parity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp + fp) / (tp + fp + tn + fn)

def NPV(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn)

# Matthews correlation coefficient
def MCC(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return ((tp*tn)-(fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

# false omission rate
def FOR(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tn)

# false discovery rate
def FDR(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (tp + fp)

# positive likelihood ratio
def PLR(y_true, y_pred):
    return sensitivity(y_true, y_pred) / (1-specificity(y_true, y_pred))

# negative likelihood ratio
def NLR(y_true, y_pred):
    return (1-sensitivity(y_true, y_pred)) / specificity(y_true, y_pred)

# diagnostic odds ratio
def DOR(y_true, y_pred):
    return PLR(y_true, y_pred) / NLR(y_true, y_pred)

# brier score
def brier_score(y_true, y_prob):
    return brier_score_loss(y_true, y_prob[:, 1])

# brier score
def log_score(y_true, y_prob):
    return log_loss(y_true, y_prob[:, 1])


prop_parity_difference = make_derived_metric(metric=prop_parity, transform="difference")
prop_parity_ratio = make_derived_metric(metric=prop_parity, transform="ratio")

predictive_rate_parity_difference = make_derived_metric(metric=precision_score, transform="difference")
predictive_rate_parity_ratio = make_derived_metric(metric=precision_score, transform="ratio")

sens_parity_difference = make_derived_metric(metric=sensitivity, transform="difference")
sens_parity_ratio = make_derived_metric(metric=sensitivity, transform="ratio")

npv_parity_difference = make_derived_metric(metric=NPV, transform="difference")
npv_parity_ratio = make_derived_metric(metric=NPV, transform="ratio")

auroc_parity_difference = make_derived_metric(metric=roc_auc_score, transform="difference")
auroc_parity_ratio = make_derived_metric(metric=roc_auc_score, transform="ratio")

mcc_parity_difference = make_derived_metric(metric=MCC, transform="difference")
mcc_parity_ratio = make_derived_metric(metric=MCC, transform="ratio")

FOR_parity_difference = make_derived_metric(metric=FOR, transform="difference")
FOR_parity_ratio = make_derived_metric(metric=FOR, transform="ratio")

FDR_parity_difference = make_derived_metric(metric=FDR, transform="difference")
FDR_parity_ratio = make_derived_metric(metric=FDR, transform="ratio")

PLR_parity_difference = make_derived_metric(metric=PLR, transform="difference")
PLR_parity_ratio = make_derived_metric(metric=PLR, transform="ratio")

NLR_parity_difference = make_derived_metric(metric=NLR, transform="difference")
NLR_parity_ratio = make_derived_metric(metric=NLR, transform="ratio")

DOR_parity_difference = make_derived_metric(metric=DOR, transform="difference")
DOR_parity_ratio = make_derived_metric(metric=DOR, transform="ratio")

ECE_parity_difference = make_derived_metric(metric=ECE, transform="difference")
ECE_parity_ratio = make_derived_metric(metric=ECE, transform="ratio")

classwise_ECE_parity_difference = make_derived_metric(metric=classwise_ECE, transform="difference")
classwise_ECE_parity_ratio = make_derived_metric(metric=classwise_ECE, transform="ratio")

MCE_parity_difference = make_derived_metric(metric=MCE, transform="difference")
MCE_parity_ratio = make_derived_metric(metric=MCE, transform="ratio")

brier_score_parity_difference = make_derived_metric(metric=brier_score, transform="difference")
brier_score_parity_ratio = make_derived_metric(metric=brier_score, transform="ratio")

log_score_parity_difference = make_derived_metric(metric=log_score, transform="difference")
log_score_parity_ratio = make_derived_metric(metric=log_score, transform="ratio")

def get_fairness_lookup():
    return ({
    'DP': [demographic_parity_difference, demographic_parity_ratio, "discrimination"],
    # 'Proportional Parity': [prop_parity_difference, prop_parity_ratio],
    'EOP': [sens_parity_difference, sens_parity_ratio, "discrimination"],
    'SPEC': [true_negative_rate_difference, true_negative_rate_ratio, "discrimination"],
    'FPR': [false_positive_rate_difference, false_positive_rate_ratio, "discrimination"],
    'FNR': [false_negative_rate_difference, false_negative_rate_ratio, "discrimination"],
    'EOD': [equalized_odds_difference, equalized_odds_ratio, "discrimination"],
    'PPV': [predictive_rate_parity_difference, predictive_rate_parity_ratio, "discrimination"],
    'NPV': [npv_parity_difference, npv_parity_ratio, "discrimination"],
    'ACC': [accuracy_score_difference, accuracy_score_ratio, "discrimination"],
    'MCC': [mcc_parity_difference, mcc_parity_ratio, "discrimination"],
    # 'FOR Parity': [FOR_parity_difference, FOR_parity_ratio, "discrimination"],
    # 'FDR Parity': [FDR_parity_difference, FDR_parity_ratio, "discrimination"],
    # 'PLR Parity': [PLR_parity_difference, PLR_parity_ratio, "discrimination"],
    # 'NLR Parity': [NLR_parity_difference, NLR_parity_ratio, "discrimination"],
    # 'DOR Parity': [DOR_parity_difference, DOR_parity_ratio, "discrimination"],
    'AUROC': [auroc_parity_difference, auroc_parity_ratio, "discrimination"],
    'ECE': [ECE_parity_difference, ECE_parity_ratio, "callibration"],
    # 'cwECE Parity': [classwise_ECE_parity_difference, classwise_ECE_parity_ratio, "callibration"],
    # 'MCE': [MCE_parity_difference, MCE_parity_ratio, "callibration"],
    'BS': [brier_score_parity_difference, brier_score_parity_ratio,"proper scoring"],
    # 'Log Score': [log_score_parity_difference, log_score_parity_ratio,"proper scoring"]
    })

