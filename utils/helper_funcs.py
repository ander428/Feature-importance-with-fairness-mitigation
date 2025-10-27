import numpy as np
import pandas as pd
from scipy import stats
import torch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch_model import SimpleNN, device, SimpleNNWrapper
import xgboost as xgb
from sklearn.cluster import KMeans

from compas import COMPASDataset

import shap
import pickle as pkl
from copy import deepcopy
import os
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.metrics import equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient

from plotnine import *
import matplotlib.pyplot as plt
import multiprocessing
from IPython.display import HTML
from tqdm.notebook import tqdm


# Load datasets
def load_datasets():
    data = COMPASDataset().df
    COMPAS = {}
    COMPAS['data'] = data[['sex', 'age_cat', 'race', 'priors_count', 'juv_misd_count', 'juv_fel_count']].dropna()
    COMPAS['data']['race'] = COMPAS['data']['race'].values == 'African-American'
    COMPAS['data'] = pd.get_dummies(COMPAS['data'])
    COMPAS['target'] = data['score_factor'][COMPAS['data'].index] == 'HighScore'
    
    data = pd.read_csv("data/afib_revised_082025_scaled_train_set_block10.csv")
    # data = pd.read_csv("data/afib_revised_082025.csv")

    AFIB = {}
    data = data.rename(columns={'Black_Race_1.0': 'race'})
    data = data.drop(["White_Race_0.0","White_Race_1.0","Black_Race_0.0","Asian_Race_0.0","Asian_Race_1.0",
               "Other_Races_0.0","Other_Races_1.0","EthnicGroup_0.0","EthnicGroup_1.0"], axis=1)
    data['race'] = data['race'].round().astype(np.float32) # force race to be binary
    AFIB['data'] = data.drop(['MBE_1year'], axis=1)
    AFIB['target'] = data.loc[:, 'MBE_1year']

    data = pd.read_csv("data/UTI Calc - Derivation dataset 2023Mar06.csv").drop('seqno', axis=1).dropna()
    UTI = {}
    UTI['data'] = data.loc[:,["agemolt12","maxtempanywherege39","History of UTI","Uncircumcised (Female or uncircumcised Male)",
                              "nosourceyn","fever_duration_hrsge48","nonblack"]]
    UTI['data']['race'] = UTI['data']['nonblack'] == 0
    UTI['data'] = UTI['data'].drop('nonblack', axis=1)
    UTI['target'] = data.loc[:,"UTI - alternative definition"]

    datasets = [(COMPAS, "COMPAS"), (AFIB, "AFIB"), (UTI, "UTI")]

    # alternate outputs for testing
    # datasets = [(COMPAS, "COMPAS"),(UTI, "UTI")]
    # datasets = [(UTI, "UTI")]
    return datasets



# Initialize models
def initialize_models(nn_params):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "Random Forest": RandomForestClassifier(max_depth=20, min_samples_split=5, min_samples_leaf=5, class_weight='balanced', n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', max_depth=20, colsample_bytree=0.5, n_jobs=-1),
        "Neural Network": SimpleNNWrapper(**nn_params)
    }
    return models

# load saved models
def load_models(dataset_name, cr=0):
    fair_st = '_fair' if cr else ''
    nn = pkl.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'Neural Network_{dataset_name}{fair_st}.pkl'), 'rb'))
    if cr:
        nn.estimator.set_device(device)
    else:
        nn.set_device(device)

    models = {
        "Logistic Regression": pkl.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'Logistic Regression_{dataset_name}{fair_st}.pkl'), 'rb')),
        "Random Forest": pkl.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'Random Forest_{dataset_name}{fair_st}.pkl'), 'rb')),
        "XGBoost": pkl.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'XGBoost_{dataset_name}{fair_st}.pkl'), 'rb')),
        "Neural Network": nn
    }

    return models

def load_train_test(dataset_name, cr=0):
    fair_st = '_fair' if cr else ''
    X_train = pkl.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'{dataset_name}_X_train{fair_st}.pkl'), 'rb'))
    X_test = pkl.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'{dataset_name}_X_test{fair_st}.pkl'), 'rb'))
    y_train = pkl.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'{dataset_name}_y_test.pkl'), 'rb'))
    y_test = pkl.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'{dataset_name}_y_test.pkl'), 'rb'))

    return X_train, X_test, y_train, y_test

def expgrad_predict_proba(expgrad_model, X):
    base_models = expgrad_model.predictors_
    weights = expgrad_model.weights_

    probs_list = [model.predict_proba(X) for model in base_models]
    weighted_probs = np.average(probs_list, axis=0, weights=weights)
    return weighted_probs

def compute_shap(model, X_train, X_test, batch_size=10):
    kmeans = KMeans(n_clusters=100).fit(X_train)

    if isinstance(model, SimpleNNWrapper):
        X_tensor = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        explainer = shap.DeepExplainer(model.model, X_tensor)

        shap_values_list = []
        for i in tqdm(range(0, X_test.shape[0], batch_size)):
            batch = X_test[i:i+batch_size]
            batch_numeric = batch.astype(np.float32)
            batch_tensor = torch.tensor(batch_numeric.values, dtype=torch.float32, device=device)
            batch_shap_values = explainer.shap_values(batch_tensor)
            shap_values_list.append(batch_shap_values)

        return np.concatenate(shap_values_list, axis=0).squeeze()
    
    elif isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
        explainer = shap.GPUTreeExplainer(
            model,
            data=kmeans.cluster_centers_,
            feature_perturbation='interventional',
            model_output='probability'
        )

        shap_values_list = []
        for i in tqdm(range(0, X_test.shape[0], batch_size)):
            batch = X_test.iloc[i:i + batch_size]
            batch_shap_values = explainer.shap_values(batch)

            # Handle binary vs. multi-class models safely
            if isinstance(batch_shap_values, list):
                # Take positive class for binary, or keep all for multiclass
                batch_shap_values = batch_shap_values[1] if len(batch_shap_values) == 2 else batch_shap_values
            shap_values_list.append(batch_shap_values)

        shap_values = np.concatenate(shap_values_list, axis=0)
        return shap_values
    else:
        pred_func = lambda x: model.predict_proba(x)[:, 1]
        if isinstance(model, ExponentiatedGradient):
            pred_func = lambda x: expgrad_predict_proba(model, x)[:, 1]
        explainer = shap.KernelExplainer(model=pred_func, data=kmeans.cluster_centers_, feature_names=X_train.columns)
        return explainer.shap_values(X_test)

# Function to compute the average SHAP values per feature
def get_avg_shap_values(shap_values, X_test):
    # Take the absolute value of SHAP values (to show magnitude of effect)
    avg_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Create a DataFrame to hold the feature names and their average SHAP values
    shap_values_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Avg_SHAP': avg_shap_values
    })

    return shap_values_df

def run_analysis(models, X_train, X_test, y_test, cr=0, dataset_name=""):
    all_shap_values = {}
    eod = []
    for model_name, model in models.items():
        pred_func = model.predict

        print(f"Processing model: {model_name}")

        # Compute SHAP explanations
        shap_explanations = compute_shap(model, X_train, X_test)
        print(f"Completed: {model_name}")

        # Compute Equalized Odds Difference
        sf = X_test['race']
        if isinstance(model, SimpleNNWrapper):
            X_test = torch.tensor(X_test.astype(np.float32).values, dtype=torch.float32, device=device)

        eod.append(equalized_odds_difference(y_test, model.predict(X_test), sensitive_features=sf))

        # Store results
        all_shap_values[model_name] = shap_explanations

    return all_shap_values, eod

# Structure output of shap results
def evaluate_models(datasets):
    all_dataset_results = {}

    for dataset_name in datasets:
        models = load_models(dataset_name)
        models_cr = load_models(dataset_name, cr=1)
        
        all_dataset_results[dataset_name] = {'shap_values': [], 'shap_data': []}

        X_train, X_test, y_train, y_test = load_train_test(dataset_name)
        # cr = pkl.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f'{dataset_name}_cr.pkl'))
                
        print(f"Calculating SHAP values for {dataset_name}...")
        shap_baseline, eod_baseline = run_analysis(models, X_train, X_test, y_test)
        shap_fair, eod_fair = run_analysis(models_cr, X_train, X_test, y_test, cr=1, dataset_name=dataset_name)


        all_dataset_results[dataset_name]['shap_values'].append(shap_baseline)
        all_dataset_results[dataset_name]['shap_data'].append(X_test)
        all_dataset_results[dataset_name]['shap_values'].append(shap_fair)
        all_dataset_results[dataset_name]['shap_data'].append(X_test)
        all_dataset_results[dataset_name]['EOD'] = {'Baseline': eod_baseline, 'Mitigated': eod_fair}
    return all_dataset_results

def wilcoxon_across_df(df):
    p_vals = {}
    for col in df.columns:
        try:
            _, p_val = stats.wilcoxon(df[col])
        except:
            p_val = 1
        p_vals[col] = p_val
    return p_vals


def plot_results(result_df, dataset_name):
    plot_data = result_df.T.reset_index().copy()
    plot_data.columns = ['_'.join(col) for col in plot_data.columns]
    plot_data_melted = plot_data.reset_index().melt(id_vars=['index_'], value_vars=plot_data.columns)
    plot_data_melted.columns = ['Feature', 'Model_Race', 'Coefficient']
    # Extract 'Model' and 'Race' from 'Model_Race'
    plot_data_melted[['Model', 'Race']] = plot_data_melted['Model_Race'].str.split('_', expand=True)

    plt.figure(figsize=(12, 8))  # Width=12, Height=8
    return (
    ggplot(plot_data_melted, aes(x='Feature', y='Coefficient', fill='Race')) +
        geom_bar(stat='identity', position='dodge') +
        facet_wrap("~Model") +
        coord_flip() +
        labs(title=f"Difference in SHAP values after \nfairness pre-processing for {dataset_name}", y=r"SHAP difference ($SHAP_{fair} - SHAP_{base}$)")
    )

def process_dataset_results(dataset_results, PRINT=False):
    def format_df(df, cohort):
        p_values_b, p_values_o =  df.groupby(cohort).apply(wilcoxon_across_df)

        df_formatted = df.groupby(cohort).mean()
        for col in df_formatted.columns:
            p_val_row = [p_values_b[col], p_values_o[col]]
            
            df_formatted[col] = [f"<b>{v:.4f}<br>({p_val:.4f})</b>" if p_val < 0.01 else f"{v:.4f}<br>({p_val:.4f})" for v, p_val in zip(df_formatted[col], p_val_row)]

        return df_formatted

    # call data by value not reference
    temp_data_results = deepcopy(dataset_results['shap_values'])
    temp_data = deepcopy(dataset_results['shap_data'])
    temp_data[0].loc[:,'race'] = ["Black" if race else "Other" for race in temp_data[0]['race']] # add labels to race variable
    temp_data[1].loc[:,'race'] = ["Black" if race else "Other" for race in temp_data[1]['race']] # add labels to race variable

    index_tuples = []
    processed_results = pd.DataFrame()
    processed_results_formatted = pd.DataFrame()

    for model in list(temp_data_results[0].keys()):
        index_tuples.append((model, 'Black'))
        index_tuples.append((model, 'Other'))
        temp = pd.DataFrame(temp_data_results[1][model]-temp_data_results[0][model], columns=temp_data[0].columns.values)
        
        groupd_df = temp.groupby(temp_data[0]['race']).mean() # use base data race information over transformed data
        grouped_df_formatted = format_df(temp, temp_data[0]['race'])

        processed_results = pd.concat([processed_results, groupd_df])
        processed_results_formatted = pd.concat([processed_results_formatted, grouped_df_formatted])

    processed_index = pd.MultiIndex.from_tuples(
        index_tuples,
        names=['Model', 'Race']
    )

    processed_results.index = processed_index
    processed_results_formatted.index = processed_index

    if PRINT:
        return HTML(processed_results_formatted.to_html(escape=0))
    else:
        return processed_results, processed_results_formatted, temp_data