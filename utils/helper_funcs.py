import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch_model import SimpleNN
import xgboost as xgb
import shap
from compas import COMPASDataset
from fairlearn.datasets import fetch_diabetes_hospital
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.metrics import equalized_odds_difference
from copy import deepcopy
from scipy import stats
from plotnine import *
import matplotlib.pyplot as plt
import multiprocessing
from IPython.display import HTML

# Load datasets
def load_datasets():
    data = COMPASDataset().df
    COMPAS = {}
    COMPAS['data'] = data[['sex', 'age_cat', 'race', 'priors_count', 'juv_misd_count', 'juv_fel_count']].dropna()
    COMPAS['data']['race'] = COMPAS['data']['race'].values == 'African-American'
    COMPAS['data'] = pd.get_dummies(COMPAS['data'])
    COMPAS['target'] = data['score_factor'][COMPAS['data'].index] == 'HighScore'
    
    # data = fetch_diabetes_hospital()
    # diabetes = {}
    # diabetes['data'] = data.data[["race", "gender", "age", "time_in_hospital", "had_inpatient_days", "medicare", "insulin", "had_emergency"]].loc[data.data['primary_diagnosis'] == "Diabetes"]
    # diabetes['data']['race'] = diabetes['data']['race'].values == "AfricanAmerican"
    # diabetes['data'] = pd.get_dummies(diabetes['data'])
    # diabetes['data'] = diabetes['data'].drop(["had_inpatient_days_False", "medicare_False", "had_emergency_False"], axis=1)
    # diabetes['data'].columns = diabetes['data'].columns.str.replace('_True', '', regex=False)
    # diabetes['target'] = data.target

    data = pd.read_csv("data/UTI Calc - Derivation dataset 2023Mar06.csv").drop('seqno', axis=1).dropna()
    UTI = {}
    UTI['data'] = data.loc[:,["agemolt12","maxtempanywherege39","History of UTI","Uncircumcised (Female or uncircumcised Male)","nosourceyn","fever_duration_hrsge48","nonblack"]]
    UTI['data']['race'] = UTI['data']['nonblack'] == 0
    UTI['data'] = UTI['data'].drop('nonblack', axis=1)
    UTI['target'] = data.loc[:,"UTI - alternative definition"]

    # datasets = [(COMPAS, "COMPAS"), (diabetes, "Diabetes"), (UTI, "UTI")]

    # alternate outputs for testing
    datasets = [(COMPAS, "COMPAS"),(UTI, "UTI")]
    # datasets = [(UTI, "UTI")]
    return datasets



# Initialize models
def initialize_models(nn_params):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss'),
        "Neural Network": SimpleNN(*nn_params)
    }
    return models

# Function to explain the model using SHAP and return the SHAP values as a DataFrame
def explain_shap(model, X_train, X_test, model_name, dataset_name):
    print(f"Explaining {model_name} on {dataset_name}...")

    shap_values_df = pd.DataFrame()

    # Create a SHAP explainer
    if model_name == "XGBoost" or model_name == "Random Forest":
        # For tree-based models (XGBoost, Random Forest)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Since it's binary classification, shap_values will be a list of 2 arrays (one for each class)
        # We focus on the SHAP values for class 1 (the positive class)
        try:
            shap_values = shap_values[:,:,1]  # Class 1 in binary classification
            base = explainer.expected_value[1]
        except:
            base = explainer.expected_value
            pass

        # Convert the SHAP values into a DataFrame
        shap_explanation = shap.Explanation(
            values=shap_values,  # SHAP values for class 1 (positive class)
            data=X_test,                  # The input features (the original data used for prediction)
            feature_names=[f'Feature {i}' for i in range(X_test.shape[1])],  # Feature names (from the dataset)
            base_values=base,  # Expected value for class 1
        )

    elif model_name == "Logistic Regression" or model_name == "Neural Network":
        # For Logistic Regression and Neural Networks, use KernelExplainer
        explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_train, 50))
        shap_values = explainer.shap_values(X_test)

        # For binary classification, shap_values will be a list of length 2
        # We focus on the SHAP values for class 1 (the positive class)
        shap_values = shap_values[:,:,1]  # Class 1 in binary classification

        # Convert the SHAP values into a DataFrame
        # shap_values_df = pd.DataFrame(shap_values, columns=[f'Feature {i}' for i in range(X_test.shape[1])])

        shap_explanation = shap.Explanation(
            values=shap_values,  # SHAP values for class 1 (positive class)
            data=X_test,                  # The input features (the original data used for prediction)
            feature_names=[f'Feature {i}' for i in range(X_test.shape[1])],  # Feature names (from the dataset)
            base_values=explainer.expected_value[1],  # Expected value for class 1
        )

    return shap_explanation

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

# Train models, evaluate, and get SHAP values
def train_and_evaluate(datasets):
    models = initialize_models()

    all_dataset_results = {}

    for data, dataset_name in datasets:
        all_dataset_results[dataset_name] = {'shap_values': [], 'shap_data': []}
        all_shap_values = {}
        fair_shap_values = {}
        try:
            X = data.data
            y = data.target
        except:
            X = data['data']
            y = data['target']

        # Sample a smaller subset of the data (use only 50 samples for faster processing)
        X_small = X[:1500]
        y_small = y[:1500]

        # Split the dataset into training and testing sets (keep it small)
        X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.3, random_state=42)

        # fairness preprocessing
        cols = list(X_train.columns)
        cols.remove("race")
        cr = CorrelationRemover(sensitive_feature_ids=['race'], alpha=0.75)
        cr.fit(X_train)
        X_train_cr = pd.DataFrame(cr.transform(X_train), columns=cols)
        X_train_cr['race'] = X_train['race'].values
        X_train_cr = X_train_cr[X_train.columns]
        X_test_cr = pd.DataFrame(cr.transform(X_test), columns=cols)
        X_test_cr['race'] = X_test['race'].values
        X_test_cr = X_test_cr[X_test.columns]

        print(f"X_train shape: {X_train.shape}")
        print(f"X_train_cr shape: {X_train_cr.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"X_test_cr shape: {X_test_cr.shape}")

        X_train, X_test, y_train, y_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)
        X_train_cr, X_test_cr = X_train_cr.reset_index(drop=True), X_test_cr.reset_index(drop=True)

        print(f"Calculating SHAP values for {dataset_name}...")
        all_shap_values, fair_shap_values, eod_values = parallel_shap_computation(models, X_train, y_train, X_train_cr, X_test, X_test_cr, dataset_name)

        all_dataset_results[dataset_name]['shap_values'].append(all_shap_values)
        all_dataset_results[dataset_name]['shap_data'].append(X_test)
        all_dataset_results[dataset_name]['shap_values'].append(fair_shap_values)
        all_dataset_results[dataset_name]['shap_data'].append(X_test_cr)
        all_dataset_results[dataset_name]['EOD'] = eod_values
    return all_dataset_results

# Function to train the model, compute SHAP values, and return the results
def process_model(model_name, model, X_train, y_train, X_train_cr, X_test, X_test_cr, dataset_name, progress_queue):
    try:
        # Train the original model
        model_cr = deepcopy(model)
        model.fit(X_train, y_train)
        model_cr.fit(X_train_cr.values, y_train)

        # Get SHAP values for both models
        shap_explanations = explain_shap(model, X_train, X_test, model_name, dataset_name)
        progress_queue.put(f"Completed: {model_name} base")
        fair_shap_explanations = explain_shap(model_cr, X_train_cr, X_test_cr, model_name, dataset_name)
        progress_queue.put(f"Completed: {model_name} fair")

        eod_baseline = equalized_odds_difference(y_train, model.predict(X_train), sensitive_features=X_train['race'])
        eod_mitigated = equalized_odds_difference(y_train, model_cr.predict(X_train), sensitive_features=X_train['race'])
        eod = {"Baseline": eod_baseline, "Mitigated": eod_mitigated}

        # Return the results
        return (model_name, shap_explanations, fair_shap_explanations, eod)
    
    except Exception as e:
        progress_queue.put(f"Error in {model_name}: {str(e)}")
        return (model_name, None, None, None)

# Function to initialize multiprocessing and collect results
def parallel_shap_computation(models, X_train, y_train, X_train_cr, X_test, X_test_cr, dataset_name):
    # Create a Manager and Queue to handle progress updates
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    # Create a Pool of processes and parallelize the computation
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Map the process_model function to each model
        results = pool.starmap(process_model, 
                               [(model_name, model, X_train, y_train, X_train_cr, X_test, X_test_cr, dataset_name, progress_queue) 
                                for model_name, model in models.items()])

    # Collect the results and handle progress updates
    all_shap_values = {}
    fair_shap_values = {}
    eod_values = {}

    # Get progress updates while collecting the results
    for model_name, shap_explanation, fair_shap_explanation, EOD in results:
        if shap_explanation is not None and fair_shap_explanation is not None and EOD is not None:
            all_shap_values[model_name] = shap_explanation
            fair_shap_values[model_name] = fair_shap_explanation
            eod_values[model_name] = EOD

    # Process the progress messages from the queue
    while not progress_queue.empty():
        print(progress_queue.get())

    return all_shap_values, fair_shap_values, eod_values

def wilcoxon_across_df(df):
    p_vals = {}
    for col in df.columns:
        try:
            _, p_val = stats.wilcoxon(df[col])
        except:
            p_val = 1
        p_vals[col] = p_val
    return p_vals

def format_df(df, cohort):
    p_values_b, p_values_o =  df.groupby(cohort).apply(wilcoxon_across_df)

    df_formatted = df.groupby(cohort).mean()
    for col in df_formatted.columns:
        p_val_row = [p_values_b[col], p_values_o[col]]
        
        df_formatted[col] = [f"<b>{v:.4f}<br>({p_val:.4f})</b>" if p_val < 0.01 else f"{v:.4f}<br>({p_val:.4f})" for v, p_val in zip(df_formatted[col], p_val_row)]

    return df_formatted


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
        temp = pd.DataFrame(temp_data_results[1][model].values-temp_data_results[0][model].values, columns=temp_data[0].columns.values)
        
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