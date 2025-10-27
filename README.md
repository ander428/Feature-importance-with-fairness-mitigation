## Description
Fairness mitigations are critical for ensuring equitable outcomes in machine learning models. However, they can significantly impact the interpretability of these models by altering feature importance. This paper investigates the effects of fairness mitigations on feature importance using SHAP (SHapley Additive exPlanations) values. The study spans three datasets—COMPAS, AFIB, and a UTI dataset—and multiple model types. It examines the impacts of achieving fairness on interpretability. The results reveal that fairness mitigations lead to significant shifts in feature importance, with variations observed across datasets and models. These findings have implications for the design and deployment of machine learning systems, especially for addressing fairness problems, in critical domains 

Folders summary:
- output: figure files and results object that are output from analysis

Data files were not included since data access is under an IRB.

## Prerequisites
- sklearn
- pandas
- numpy
- xgboost
- shap
- fairlearn
- copy
- scipy
- plotnine
- matplotlib
- multiprocessing
- IPython
- pickle
- imblearn

## Authors
Joshua Anderson: jwa45@pitt.edu

## Acknowledgments
Thank you to UPMC Department of Pediatrics for providing the UTI dataset.

