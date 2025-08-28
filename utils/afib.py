# block1 code
import pandas as pd
import numpy as np

def preprocess_data(input_file_path, output_file_path):
    # Load the dataset, treating "NA" as missing values
    df = pd.read_csv(input_file_path, na_values='NA', low_memory=False)

    # List of columns to keep
    columns_to_keep = [
            "s.no", "RACE", "ETHNIC_GROUP", "FEMALE", "AGE", "BMI", "BP_SYSTOLIC", "BP_DIASTOLIC", "PULSE", "INS_MC", "INS_MA", "ACTIVE_CANCER", "PH_HST", "DIAB_HST", "CAD_HST", "HYPERLIPIDEMIA_HST",
            "CARDIAC_ARREST_HST", "VTVF_HST", "LUNG_CANCER_HST", "ASTHMA_HST", "STROKE_HST",
            "HEMO_STROKE_HST", "DVT_HST", "PE_HST", "MAJOR_BLEED_HST", "OBS_SLEEPAPNEA_HST", "TIA_HST",
            "PAROXYSMAL_AF", "LONGSTNDING_AF", "OTHER_AF", "AMYLOID_HX", "SARCOIDOSIS_HX",
            "HYPERTROPHIC_CMP_HX", "CCI_MI", "CCI_PERIPHERAL_VASC", "CCI_DEMENTIA", "CCI_COPD", "CCI_RHEUMATIC_DISEASE", "CCI_PEPTIC_ULCER", "CCI_RENAL_DISEASE", "CCI_AIDS_HIV", "CCI_TOTAL_SCORE", "ELIX_CARDIAC_ARRTHYTHMIAS", "ELIX_VALVULAR_DISEASE",
            "ELIX_PULM_CIRC_DISORDERS", "ELIX_HYPERTENSION", "ELIX_PARALYSIS", "ELIX_HYPOTHYROIDISM", "ELIX_LIVER_DISEASE",
            "ELIX_LYMPHOMA", "ELIX_COAG_DEFICIENCY", "ELIX_FLUID_ELECTROLYTE_DIS", "ELIX_ANEMIA_BLOOD_LOSS", "ELIX_DEFICIENCY_ANEMIAS",
            "ELIX_PSYCHOSES", "ELIX_DEPRESSION", "TOBACCO_STATUS", "ALCOHOL_STATUS", "ILL_DRUG_STATUS", "PRE_SEVERE_MV_REGURG",
            "PRE_SEVERE_AV_STEN", "LVEF_VISIT", "LA_DIA_VISIT", "LVDD_VISIT", "LVSD_VISIT", "MR_VISIT",
            "MS_VISIT", "ASA", "CLASS_I", "CLASS_III", "CCB", "MRA", "BB", "STATIN", "INSULIN", "METFORMIN", "SGLT2_INHIBITORS",
            "DPP4_INHIBTOR", "TZD", "GLP1", "SULFONYLUREA", "AMIODARONE", "SOTALOL", "DOFETILIDE",
            "FLECAINIDE", "PROPAFENONE", "ACEI", "ARB", "ARNI", "EGFR", "CR", "CHOLESTEROL", "LDL",
            "HDL", "HEMOGLOBIN_A1C", "TRIGLYCERIDE", "TSH", "PRE_AV_NODE_ABLATION", "PRE_PVI_ABLATION",
            "PRE_PPM_IMPLANT", "PRE_ICD_IMPLANT", "PRE_CRT_D_IMPLANT",
            "PRE_CRT_P_IMPLANT", "PRE_CARDIOVERSION", "PRE_CABG", "NONCARDIAC_READMIT_FLAG",
            "NONCARDIAC_READMIT_DAYS", "CARDIAC_READMIT_FLAG", "CARDIAC_READMIT_DAYS", "STEMI_READMIT_FLAG",
            "STEMI_READMIT_DAYS", "NSTEMI_REVASC_READMIT_FLAG", "NSTEMI_REVASC_READMIT_DAYS",
            "USA_REVASC_READMIT_FLAG", "USA_REVASC_READMIT_DAYS", "SA_REVASC_READMIT_FLAG", "SA_REVASC_READMIT_DAYS", "I_STROKE_READMIT_FLAG",
            "I_STROKE_READMIT_DAYS", "H_STROKE_READMIT_FLAG", "H_STROKE_READMIT_DAYS", "TIA_READMIT_FLAG",
            "TIA_READMIT_DAYS", "MAJOR_BLEED_READMIT_FLAG", "MAJOR_BLEED_READMIT_DAYS", "CTB", "FOLLOWUP_DAYS", 
            "ALL_HF", "HFrEF", "HFmrEF", "HFpEF", "RHYTHM_EVER", "ABLATION_EVER",
            "REVASC_HST"
        ]

    # Filter the DataFrame to only include the specified columns and copy into a new dataframe so that original remains unchanged
    df_filtered = df[columns_to_keep].copy()

    # Create binary columns for race categories
    df_filtered.loc[:, 'White_Race'] = (df_filtered['RACE'] == 'White').astype(int)
    df_filtered.loc[:, 'Black_Race'] = (df_filtered['RACE'] == 'Black').astype(int)

    # Combine multiple Asian categories into a single binary column
    asian_races = ['Chinese', 'Other Asian', 'Korean', 'Japanese', 'Vietnamese']
    df_filtered.loc[:, 'Asian_Race'] = df_filtered['RACE'].isin(asian_races).astype(int)

    # Combine specified categories into 'Other_Races'
    other_races = ['American Indian', 'Filipino', 'Indian (Asian)', 'Samoan', 'Other', 
                   'Other Pacific Islander', 'Alaska Native', 'Hawaiian', 'Guam/Chamorro']
    df_filtered.loc[:, 'Other_Races'] = df_filtered['RACE'].isin(other_races).astype(int)

    # Handle 'Declined' and 'Not Specified' for all new race columns by setting them to NaN
    na_values = ['Declined', 'Not Specified']
    for race_column in ['White_Race', 'Black_Race', 'Asian_Race', 'Other_Races']:
        df_filtered.loc[df_filtered['RACE'].isin(na_values), race_column] = np.nan

    # Add a new column 'NON_COMORBID_CONTROLS'
    df_filtered['NON_COMORBID_CONTROLS'] = ((df_filtered['CCI_TOTAL_SCORE'] == 0) & (df_filtered['MAJOR_BLEED_HST'] == 0)).astype(int)

    # Dropping the columns 'RACE'
    df_filtered.drop(columns=['RACE'], inplace=True)

    # Save the preprocessed DataFrame to a CSV file
    df_filtered.to_csv(output_file_path, index=False)

    # Return the preprocessed DataFrame
    return df_filtered


# block2 code
import pandas as pd
import numpy as np

def calculate_clinical_outcomes(df_filtered, outcome, timepoint):
    def calculate_outcome(df, time_frame, day_columns, flag_columns, outcome_prefix):
        day_limit = time_frame
        temp_df = df[day_columns + flag_columns].copy()
        temp_df[day_columns] = temp_df[day_columns].apply(lambda x: x.where(x < day_limit, other=np.nan))
        temp_df[f'min_day_{timepoint}'] = temp_df[day_columns].min(axis=1)

        df[f'{outcome_prefix}{timepoint}'] = 0
        df[f'{outcome_prefix}{timepoint}_TIME'] = day_limit

        for index, row in temp_df.iterrows():
            min_day = row[f'min_day_{timepoint}']
            if pd.notna(min_day):
                min_day = round(min_day, 1)
                min_day = int(min_day)
                min_day_col = row[day_columns].idxmin()

                if pd.notna(min_day_col):
                    flag_col_prefix = min_day_col.replace('_DAYS', '_FLAG')
                    for flag_col in flag_columns:
                        if flag_col.startswith(flag_col_prefix):
                            df.at[index, f'{outcome_prefix}{timepoint}'] = row[flag_col]
                            if row[flag_col] == 0:
                                df.at[index, f'{outcome_prefix}{timepoint}_TIME'] = int(temp_df.loc[index, day_columns].where(lambda x: x < day_limit).max())
                            else:
                                df.at[index, f'{outcome_prefix}{timepoint}_TIME'] = min_day

        df[f'{outcome_prefix}{timepoint}'] = df[f'{outcome_prefix}{timepoint}'].astype(int)
        df[f'{outcome_prefix}{timepoint}_TIME'] = df[f'{outcome_prefix}{timepoint}_TIME'].fillna(day_limit).astype(int)

    time_frames = {'1year': 365, '2year': 730, '5year': 1825}
    day_limit = time_frames[timepoint]

    if outcome == 'STEMIOrRevascforCAD':
        cad_days_columns = ['STEMI_READMIT_DAYS', 'NSTEMI_REVASC_READMIT_DAYS', 'USA_REVASC_READMIT_DAYS', 'SA_REVASC_READMIT_DAYS']
        cad_flag_columns = ['STEMI_READMIT_FLAG', 'NSTEMI_REVASC_READMIT_FLAG', 'USA_REVASC_READMIT_FLAG', 'SA_REVASC_READMIT_FLAG']
        calculate_outcome(df_filtered, day_limit, cad_days_columns, cad_flag_columns, 'STEMIOrRevascforCAD')

    elif outcome == 'IstrokeOrTIAOrSTEMIOrRevascforCAD':
        thrombotic_days_columns = [f'STEMIOrRevascforCAD{timepoint}_TIME'] + ['I_STROKE_READMIT_DAYS', 'TIA_READMIT_DAYS']
        thrombotic_flag_columns = [f'STEMIOrRevascforCAD{timepoint}'] + ['I_STROKE_READMIT_FLAG', 'TIA_READMIT_FLAG']
        calculate_outcome(df_filtered, day_limit, thrombotic_days_columns, thrombotic_flag_columns, 'IstrokeOrTIAOrSTEMIOrRevascforCAD')

    elif outcome == 'MBE':
        df_filtered[f'MBE_{timepoint}'] = df_filtered.apply(
            lambda row: 1 if (row['MAJOR_BLEED_READMIT_DAYS'] < day_limit and row['MAJOR_BLEED_READMIT_FLAG'] == 1) else 0,
            axis=1
        )
        df_filtered[f'MBE_{timepoint}_TIME'] = df_filtered.apply(
            lambda row: row['MAJOR_BLEED_READMIT_DAYS'] if row['MAJOR_BLEED_READMIT_DAYS'] < day_limit else day_limit,
            axis=1
        )
        df_filtered[f'MBE_{timepoint}'] = df_filtered[f'MBE_{timepoint}'].astype(int)
        df_filtered[f'MBE_{timepoint}_TIME'] = df_filtered[f'MBE_{timepoint}_TIME'].astype(int)

    elif outcome == 'IstrokeOrTIA':
        df_filtered[f'IstrokeOrTIA{timepoint}'] = 0
        df_filtered[f'IstrokeOrTIA{timepoint}_TIME'] = day_limit
        for index, row in df_filtered.iterrows():
            min_days = min([day for day in [row['I_STROKE_READMIT_DAYS'], row['TIA_READMIT_DAYS']] if day < day_limit], default=day_limit)
            df_filtered.at[index, f'IstrokeOrTIA{timepoint}_TIME'] = int(min_days)
            if min_days == row['I_STROKE_READMIT_DAYS']:
                df_filtered.at[index, f'IstrokeOrTIA{timepoint}'] = row['I_STROKE_READMIT_FLAG']
            elif min_days == row['TIA_READMIT_DAYS']:
                df_filtered.at[index, f'IstrokeOrTIA{timepoint}'] = row['TIA_READMIT_FLAG']
            if df_filtered.at[index, f'IstrokeOrTIA{timepoint}'] == 0 and min_days < day_limit:
                df_filtered.at[index, f'IstrokeOrTIA{timepoint}_TIME'] = int(max([day for day in [row['I_STROKE_READMIT_DAYS'], row['TIA_READMIT_DAYS']] if day < day_limit], default=day_limit))
        df_filtered[f'IstrokeOrTIA{timepoint}'] = df_filtered[f'IstrokeOrTIA{timepoint}'].astype(int)
        df_filtered[f'IstrokeOrTIA{timepoint}_TIME'] = df_filtered[f'IstrokeOrTIA{timepoint}_TIME'].astype(int)

    elif outcome in ['H_STROKE_READMIT', 'Mortality']:
        if outcome == 'Mortality':
            df_filtered[f'{outcome}_{timepoint}_TIME'] = df_filtered['FOLLOWUP_DAYS'].apply(lambda x: x if x < day_limit else day_limit)
        else:
            df_filtered[f'{outcome}_{timepoint}_TIME'] = day_limit
        flag_column = 'H_STROKE_READMIT_FLAG' if outcome == 'H_STROKE_READMIT' else 'CTB'
        df_filtered[f'{outcome}_{timepoint}'] = df_filtered.apply(
            lambda row: 1 if (row['FOLLOWUP_DAYS' if outcome == 'Mortality' else 'H_STROKE_READMIT_DAYS'] < day_limit and row[flag_column] == 1) else 0, axis=1
        )
        if outcome != 'Mortality':
            df_filtered[f'{outcome}_{timepoint}_TIME'] = df_filtered.apply(
                lambda row: row['H_STROKE_READMIT_DAYS'] if row['H_STROKE_READMIT_DAYS'] < day_limit else day_limit, axis=1
            )

    return df_filtered

# block3 code
# Stratified splitting based on Sex and Black race of the dataset into training (70%) and test (non-comorbid controls and random controls)
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

MY_RANDOM_STATE = 42

def split_data(df_filtered, outcome, timepoint, stratify_cols, train_pct, non_comorbid_pct, random_pct):
    # Define the outcome variable
    outcome_var = f'{outcome}_{timepoint}'

    # Define stratification column based on the provided factors
    df_filtered['stratify_col'] = df_filtered[stratify_cols].apply(lambda row: '_'.join(row.astype(str)), axis=1)

    # Prepare for stratified split
    if outcome_var in df_filtered.columns and 'stratify_col' in df_filtered.columns:
        df_stratify = df_filtered.dropna(subset=['stratify_col', outcome_var])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_pct), random_state=MY_RANDOM_STATE)
        for train_index, test_index in sss.split(df_stratify, df_stratify['stratify_col']):
            train_set = df_stratify.iloc[train_index]
            test_set = df_stratify.iloc[test_index]
    else:
        print("Error: The outcome variable or the stratification column does not exist in the DataFrame.")
        return None, None, None

    # Split test set into non-comorbid and random subsets
    if 'NON_COMORBID_CONTROLS' in test_set.columns:
        test_set_non_comorbid = test_set[test_set['NON_COMORBID_CONTROLS'] == 1].sample(frac=non_comorbid_pct, random_state=MY_RANDOM_STATE)
        test_set_random = test_set.drop(test_set_non_comorbid.index).sample(frac=random_pct, random_state=MY_RANDOM_STATE)
    else:
        print("'NON_COMORBID_CONTROLS' column not found.")
        test_set_non_comorbid = pd.DataFrame()
        test_set_random = test_set.sample(frac=random_pct, random_state=MY_RANDOM_STATE)

    # Check the distribution of the outcome variable in each set
    print(f"Distribution of {outcome_var} in the training set:")
    print(train_set[outcome_var].value_counts(normalize=True))
    print(f"\nDistribution of {outcome_var} in the non-comorbid test set:")
    print(test_set_non_comorbid[outcome_var].value_counts(normalize=True))
    print(f"\nDistribution of {outcome_var} in the random test set:")
    print(test_set_random[outcome_var].value_counts(normalize=True))

    return train_set, test_set_non_comorbid, test_set_random

# block4 code
import pandas as pd
import numpy as np

def classify_columns_filtered(df_filtered):
    categorical = []
    continuous = []

    # Threshold for the number of unique values to consider a column categorical
    unique_threshold = 20

    for column in df_filtered.columns:
        unique_count = df_filtered[column].nunique()

        # Determine if the column is categorical based on unique values and data type
        if df_filtered[column].dtype == 'object' or unique_count <= unique_threshold:
            categorical.append(column)
        else:
            continuous.append(column)

    return categorical, continuous

def additional_preprocessing(train_set, test_set_non_comorbid, test_set_random):
    # Perform additional preprocessing on train_set, test_set_non_comorbid, and test_set_random

    def preprocess_dataset(df_filtered):
        # Ethnic group modifications
        ethnic_group_map = {
            'Not Hispanic or Latino': 0,
            'Hispanic or Latino': 1,
            'Declined': np.nan,
            'Not Specified': np.nan
        }
        df_filtered.loc[:, 'EthnicGroup'] = df_filtered['ETHNIC_GROUP'].map(ethnic_group_map)

        # Creating the EVER_SMOKER column
        df_filtered['EVER_SMOKER'] = df_filtered['TOBACCO_STATUS'].apply(lambda x: 1 if x in [1, 2, 3, 4, 5] else (0 if x == 0 else np.nan))

        # Creating the CURRENT_SMOKER column
        df_filtered['CURRENT_SMOKER'] = df_filtered['TOBACCO_STATUS'].apply(lambda x: 1 if x == 5 else (0 if x in [0, 1, 2, 3, 4] else np.nan))

        # Creating the EVER_ALCOHOL column
        df_filtered['EVER_ALCOHOL'] = df_filtered['ALCOHOL_STATUS'].apply(lambda x: 1 if x in [2, 3, 4, 5] else (0 if x == 0 else np.nan))

        # Creating the CURRENT_ALCOHOL column
        df_filtered['CURRENT_ALCOHOL'] = df_filtered['ALCOHOL_STATUS'].apply(lambda x: 1 if x == 5 else (0 if x in [0, 2, 3, 4] else np.nan))

        # Creating the EVER_ILL_DRUG column
        df_filtered['EVER_ILL_DRUG'] = df_filtered['ILL_DRUG_STATUS'].apply(lambda x: 1 if x in [2, 3, 4, 5] else (0 if x == 0 else np.nan))

        # Creating the CURRENT_ILL_DRUG column
        df_filtered['CURRENT_ILL_DRUG'] = df_filtered['ILL_DRUG_STATUS'].apply(lambda x: 1 if x == 5 else (0 if x in [0, 2, 3, 4] else np.nan))

        # Creating the MR_MODSEVERE_VISIT column
        df_filtered['MR_MODSEVERE_VISIT'] = df_filtered['MR_VISIT'].apply(lambda x: 1 if x in ['2.5+', '3+', '3.5+', '4+', 'Moderate', 'Moderately Severe'] else (0 if x in ['2+', '1+', '-', 'Trace-1+', 'Mild', 'Trace', '1+-2+', 'Trivial'] else np.nan))

        # Creating the MR_SEVERE_VISIT column
        df_filtered['MR_SEVERE_VISIT'] = df_filtered['MR_VISIT'].apply(lambda x: 1 if x in ['3+', '3.5+', '4+', 'Moderately Severe'] else (0 if x in ['2.5+', '2+', '1+', '-', 'Trace-1+', 'Mild', 'Trace', '1+-2+', 'Trivial', 'Moderate'] else np.nan))

        # Creating the MS_MODSEVERE_VISIT column
        df_filtered['MS_MODSEVERE_VISIT'] = df_filtered['MS_VISIT'].apply(lambda x: 1 if x in ['Severe', 'moderate-severe'] else (0 if x in ['Mild', 'Moderate', 'mild-moderate', 'none'] else np.nan))

        # Creating the MS_SEVERE_VISIT column
        df_filtered['MS_SEVERE_VISIT'] = df_filtered['MS_VISIT'].apply(lambda x: 1 if x == 'Severe' else (0 if x in ['Mild', 'Moderate', 'mild-moderate', 'moderate-severe', 'none'] else np.nan))

        # Convert 'True'/'False' to 1/0 for specified columns
        columns_to_convert = ['HFrEF', 'HFmrEF', 'HFpEF', 'RHYTHM_EVER', 'ABLATION_EVER']

        for column in columns_to_convert:
            df_filtered[column] = df_filtered[column].apply(lambda x: 1 if x == True else (0 if x == False else np.nan))

        # Split the 'AGE' column into specified binary columns
        df_filtered.loc[:, 'AGE_LESS_THAN_50'] = (df_filtered['AGE'] < 50).astype(int)
        df_filtered.loc[:, 'AGE_50TO64'] = ((df_filtered['AGE'] >= 50) & (df_filtered['AGE'] <= 64)).astype(int)
        df_filtered.loc[:, 'AGE_65TO74'] = ((df_filtered['AGE'] >= 65) & (df_filtered['AGE'] <= 75)).astype(int)
        df_filtered.loc[:, 'AGE_MORE_THAN_75'] = (df_filtered['AGE'] > 75).astype(int)

        # Refining columns to add more features
        def add_new_columns(df_filtered):
            required_columns = ['BP_DIASTOLIC', 'BP_SYSTOLIC', 'DIAB_HST', 'HEMOGLOBIN_A1C', 'EGFR', 'ELIX_ANEMIA_BLOOD_LOSS', 'ELIX_DEFICIENCY_ANEMIAS']
            missing_columns = [col for col in required_columns if col not in df_filtered.columns]
            
            if not missing_columns:
                mean_arterial_pressure = df_filtered['BP_DIASTOLIC'] + (df_filtered['BP_SYSTOLIC'] - df_filtered['BP_DIASTOLIC']) / 3
                
                df_filtered = df_filtered.assign(
                    Poorly_controlled_HTN_SBPmoreTHAN149=(df_filtered['BP_SYSTOLIC'] > 149).astype(int),
                    DM_poorly_controlled=((df_filtered['DIAB_HST'] == 1) & (df_filtered['HEMOGLOBIN_A1C'] > 6.5)).astype(int),
                    Prediabetes=((df_filtered['DIAB_HST'] == 0) & (df_filtered['HEMOGLOBIN_A1C'] > 5.8)).astype(int),
                    CKD_2=((df_filtered['EGFR'] >= 60) & (df_filtered['EGFR'] < 89)).astype(int),
                    CKD_3=((df_filtered['EGFR'] >= 30) & (df_filtered['EGFR'] < 59)).astype(int),
                    CKD_4=((df_filtered['EGFR'] >= 15) & (df_filtered['EGFR'] < 29)).astype(int),
                    CKD_5=(df_filtered['EGFR'] < 15).astype(int),
                    Anemia_history=((df_filtered['ELIX_ANEMIA_BLOOD_LOSS'] == 1) | (df_filtered['ELIX_DEFICIENCY_ANEMIAS'] == 1)).astype(int),
                    Obese=(df_filtered['BMI'] > 29.9).astype(int),
                    Underweight=(df_filtered['BMI'] < 18.5).astype(int),
                    HTN_gd1=((mean_arterial_pressure >= 105.68) & (mean_arterial_pressure <= 119.00)).astype(int),
                    HTN_gd2=((mean_arterial_pressure >= 119.01) & (mean_arterial_pressure <= 132.33)).astype(int),
                    HTN_gd3=(mean_arterial_pressure >= 132.34).astype(int),
                    Bradycardic=(df_filtered['PULSE'] < 60).astype(int),
                    Tachycardic=(df_filtered['PULSE'] > 100).astype(int),
                )
            else:
                print(f"Skipping new columns addition due to missing columns: {missing_columns}")
            return df_filtered

        df_filtered = add_new_columns(df_filtered)
        
        # Remove columns
        columns_to_remove = ['RACE', 'ETHNIC_GROUP', 'TOBACCO_STATUS', 'ALCOHOL_STATUS', 'ILL_DRUG_STATUS', 'MR_VISIT', 'MS_VISIT', 'BP_DIASTOLIC', 'BP_SYSTOLIC', 'HEMOGLOBIN_A1C', 'ELIX_ANEMIA_BLOOD_LOSS', 'ELIX_DEFICIENCY_ANEMIAS', 'NON_COMORBID_CONTROLS', 'stratify_col', 'CCI_TOTAL_SCORE', 'ETHNIC_GROUP', 'PULSE', 'NONCARDIAC_READMIT_DAYS', 'CARDIAC_READMIT_DAYS', 'STEMI_READMIT_DAYS', 'NSTEMI_REVASC_READMIT_DAYS', 'USA_REVASC_READMIT_DAYS', 'SA_REVASC_READMIT_DAYS', 'I_STROKE_READMIT_DAYS', 'H_STROKE_READMIT_DAYS', 'TIA_READMIT_DAYS', 'MAJOR_BLEED_READMIT_DAYS', 'FOLLOWUP_DAYS', 'BMI', 'LVEF_VISIT', 'CR', 'TSH', 'CHOLESTEROL', 'LDL', 'HDL', 'TRIGLYCERIDE']
        df_filtered = df_filtered.drop(columns=columns_to_remove, axis=1, errors='ignore')
        
        return df_filtered
    
    # Apply preprocessing to each dataset
    train_set = preprocess_dataset(train_set)
    test_set_non_comorbid = preprocess_dataset(test_set_non_comorbid)
    test_set_random = preprocess_dataset(test_set_random)
    
    # Classify columns in each dataset
    categorical_cols_train, continuous_cols_train = classify_columns_filtered(train_set)
    categorical_cols_non_comorbid, continuous_cols_non_comorbid = classify_columns_filtered(test_set_non_comorbid)
    categorical_cols_random, continuous_cols_random = classify_columns_filtered(test_set_random)

    return train_set, test_set_non_comorbid, test_set_random, categorical_cols_train, continuous_cols_train, categorical_cols_non_comorbid, continuous_cols_non_comorbid, categorical_cols_random, continuous_cols_random

    # Define the constraints for each variable
    constraints = {
        'BMI': {'lower': 14, 'upper': 60},
        'PULSE': {'lower': 40, 'upper': 150},
        'LVEF_VISIT': {'upper': 70},
        'CR': {'upper': 10.0},
        'CHOLESTEROL': {'lower': 100, 'upper': 400},
        'LDL': {'lower': 20, 'upper': 200},
        'HDL': {'lower': 20, 'upper': 140},
        'TRIGLYCERIDE': {'lower': 75, 'upper': 1200},
        'TSH': {'lower': 0.1, 'upper': 10},
    }
    
    # Function to apply constraints and rounding to continuous variables
    def apply_constraints_and_rounding(df_filtered, continuous_cols):
        for var in continuous_cols:
            if var in constraints:
                limits = constraints[var]
                if 'lower' in limits and 'upper' in limits:
                    df_filtered.loc[:, var] = df_filtered[var].clip(lower=limits['lower'], upper=limits['upper'])
                elif 'lower' in limits:
                    df_filtered.loc[:, var] = df_filtered[var].clip(lower=limits['lower'])
                elif 'upper' in limits:
                    df_filtered.loc[:, var] = df_filtered[var].clip(upper=limits['upper'])
            df_filtered.loc[:, var] = df_filtered[var].round(1)
        return df_filtered
    
    # Apply constraints and rounding to each dataset
    train_set = apply_constraints_and_rounding(train_set, continuous_cols_train)
    test_set_non_comorbid = apply_constraints_and_rounding(test_set_non_comorbid, continuous_cols_non_comorbid)
    test_set_random = apply_constraints_and_rounding(test_set_random, continuous_cols_random)

    # Return the preprocessed datasets along with the categorical and continuous columns for each dataset
    return train_set, test_set_non_comorbid, test_set_random, categorical_cols_train, continuous_cols_train, categorical_cols_non_comorbid, continuous_cols_non_comorbid, categorical_cols_random, continuous_cols_random

import pandas as pd
from sklearn.impute import SimpleImputer

def impute_missing_values(df_filtered, categorical_cols, continuous_cols):
    # Create separate DataFrames for categorical and continuous columns
    df_categorical = df_filtered[categorical_cols]
    df_continuous = df_filtered[continuous_cols]

    # Impute missing values in categorical columns using most frequent value
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df_categorical_imputed = pd.DataFrame(imputer_categorical.fit_transform(df_categorical), columns=categorical_cols, index=df_filtered.index)

    # Impute missing values in continuous columns using median
    imputer_continuous = SimpleImputer(strategy='median')
    df_continuous_imputed = pd.DataFrame(imputer_continuous.fit_transform(df_continuous), columns=continuous_cols, index=df_filtered.index)

    # Combine the imputed DataFrames
    df_filtered_imputed = pd.concat([df_categorical_imputed, df_continuous_imputed], axis=1)

    return df_filtered_imputed

def compute_risk_scores(df_filtered, categorical_cols, continuous_cols):
    # Impute missing values
    df_filtered_imputed = impute_missing_values(df_filtered, categorical_cols, continuous_cols)

    # HAS-BLED score calculation
    HASBLED_score = (
        (df_filtered_imputed['ELIX_HYPERTENSION'] == 1).astype(int) +
        (df_filtered_imputed['CCI_RENAL_DISEASE'] == 1).astype(int) +
        (df_filtered_imputed['ELIX_LIVER_DISEASE'] == 1).astype(int) +
        ((df_filtered_imputed['STROKE_HST'] == 1) | (df_filtered_imputed['TIA_HST'] == 1)).astype(int) +
        ((df_filtered_imputed['MAJOR_BLEED_HST'] == 1) | (df_filtered_imputed['ELIX_COAG_DEFICIENCY'] == 1)).astype(int) +
        ((df_filtered_imputed['AGE_65TO74'] == 1) | (df_filtered_imputed['AGE_MORE_THAN_75'] == 1)).astype(int) +
        (df_filtered_imputed['ASA'] == 1).astype(int)
    )

    # ATRIA score calculation
    ATRIA_score = (
        ((df_filtered_imputed['Anemia_history'] == 1).astype(int) * 3 +
        ((df_filtered_imputed['CKD_4'] == 1) | (df_filtered_imputed['CKD_5'] == 1)).astype(int) * 3 +
        (df_filtered_imputed['AGE_MORE_THAN_75'] == 1).astype(int) * 2 +
        (df_filtered_imputed['MAJOR_BLEED_HST'] == 1).astype(int) +
        (df_filtered_imputed['ELIX_HYPERTENSION'] == 1).astype(int)
        )
    )

    # ORBIT score calculation
    ORBIT_score = (
        ((df_filtered_imputed['Anemia_history'] == 1).astype(int) * 2 +
        ((df_filtered_imputed['CKD_3'] == 1) | (df_filtered_imputed['CKD_4'] == 1) | (df_filtered_imputed['CKD_5'] == 1)).astype(int) +
        (df_filtered_imputed['AGE_MORE_THAN_75'] == 1).astype(int) +
        (df_filtered_imputed['MAJOR_BLEED_HST'] == 1).astype(int) +
        (df_filtered_imputed['ASA'] == 1).astype(int)
        )
    )

    # CHA2DS2-VASc score calculation
    CHA2DS2_VASc_score = (
        df_filtered_imputed['ALL_HF'].astype(int) +
        (df_filtered_imputed['AGE_65TO74'] == 1).astype(int) +
        (df_filtered_imputed['AGE_MORE_THAN_75'] >= 75).astype(int) * 2 +
        df_filtered_imputed['FEMALE'].astype(int) +
        df_filtered_imputed['ELIX_HYPERTENSION'].astype(int) +
        ((df_filtered_imputed['STROKE_HST'] == 1) | (df_filtered_imputed['TIA_HST'] == 1)).astype(int) * 2 +
        ((df_filtered_imputed['CCI_MI'] == 1) | (df_filtered_imputed['CCI_PERIPHERAL_VASC'] == 1)).astype(int) +
        df_filtered_imputed['DIAB_HST'].astype(int)
    )

    # Add risk scores to the DataFrame
    df_filtered['HASBLED_score'] = HASBLED_score
    df_filtered['ATRIA_score'] = ATRIA_score
    df_filtered['ORBIT_score'] = ORBIT_score
    df_filtered['CHA2DS2_VASc_score'] = CHA2DS2_VASc_score

    return df_filtered


def add_risk_scores_to_datasets(train_set_filtered, test_set_non_comorbid_filtered, test_set_random_filtered, categorical_cols_train, continuous_cols_train, categorical_cols_non_comorbid, continuous_cols_non_comorbid, categorical_cols_random, continuous_cols_random):
    # Apply risk score computation to each dataset
    train_set_filtered = compute_risk_scores(train_set_filtered, categorical_cols_train, continuous_cols_train)
    test_set_non_comorbid_filtered = compute_risk_scores(test_set_non_comorbid_filtered, categorical_cols_non_comorbid, continuous_cols_non_comorbid)
    test_set_random_filtered = compute_risk_scores(test_set_random_filtered, categorical_cols_random, continuous_cols_random)

    # Add risk score columns to the continuous columns
    continuous_cols_train.extend(['HASBLED_score', 'ATRIA_score', 'ORBIT_score', 'CHA2DS2_VASc_score'])
    continuous_cols_non_comorbid.extend(['HASBLED_score', 'ATRIA_score', 'ORBIT_score', 'CHA2DS2_VASc_score'])
    continuous_cols_random.extend(['HASBLED_score', 'ATRIA_score', 'ORBIT_score', 'CHA2DS2_VASc_score'])

    return train_set_filtered, test_set_non_comorbid_filtered, test_set_random_filtered, continuous_cols_train, continuous_cols_non_comorbid, continuous_cols_random

#Block6 code
import pandas as pd
import numpy as np
from scipy import stats

def summary_stats(df_filtered, column, outcome, is_continuous=True):
    if is_continuous:
        median = np.median(df_filtered[column].dropna())
        q25, q75 = np.percentile(df_filtered[column].dropna(), [25, 75])
        return f"{median:.2f} (IQR {q25:.2f}-{q75:.2f})"
    else:
        total = df_filtered[column].value_counts().sum()
        if 1.0 in df_filtered[column].value_counts():
            count = df_filtered[column].value_counts()[1.0]
            percentage = (count / total) * 100
            return f"{count} ({percentage:.2f}%)"
        else:
            return "0 (0%)"

def compare_groups(df_filtered, column, outcome, is_continuous=True):
    group1 = df_filtered[df_filtered[outcome] == 1]
    group2 = df_filtered[df_filtered[outcome] == 0]
    if is_continuous:
        stat, p_val = stats.mannwhitneyu(group1[column].dropna(), group2[column].dropna())
    else:
        table = pd.crosstab(df_filtered[column], df_filtered[outcome])
        stat, p_val = stats.chi2_contingency(table)[:2]
    if p_val < 0.001:
        return "<0.001"
    else:
        return f"{p_val:.3f}"

def extrapolate_name(variable_name):
    name_mapping = {
        'AGE': 'Age (Years)',
        'AGE_LESS_THAN_50': 'Age Less Than 50 Years',
        'AGE_50TO64': 'Age 50 to 64 Years',
        'AGE_65TO74': 'Age 65 to 74 Years',
        'AGE_MORE_THAN_75': 'Age More Than 75 Years',
        'FEMALE': 'Sex (Female)',
        'White_Race': 'White Race',
        'Black_Race': 'Black Race',
        'Asian_Race': 'Asian Race',
        'Other_Races': 'Other Races',
        'EthnicGroup': 'Hispanic or Latino Ethnic Group',
        'EVER_SMOKER': 'Ever Smoker',
        'CURRENT_SMOKER': 'Current Smoker',
        'EVER_ALCOHOL': 'Ever Alcohol Use',
        'CURRENT_ALCOHOL': 'Current Alcohol Use',
        'EVER_ILL_DRUG': 'Ever Illicit Drug Use',
        'CURRENT_ILL_DRUG': 'Current Illicit Drug Use',
        'INS_MC': 'Medicare Insurance',
        'INS_MA': 'Medicaid Insurance',
        'BMI': 'Body Mass Index (kilograms per square meter)',
        'Obese': 'Obese with BMI >29.9',
        'Underweight': 'Underweight with BMI <18.5',
        'Poorly_controlled_HTN_SBPmoreTHAN149': 'Poorly Controlled Hypertension (Systolic Blood Pressure More Than 149 mmHg)',
        'Bradycardic': 'Bradycardic with heart rate <60 beats per minute',
        'Tachycardic': 'Tachycardic with heart rate >100 beats per minute',
        'ELIX_HYPERTENSION': 'Hypertension',
        'HTN_gd1': 'Grade 1 Hypertension',
        'HTN_gd2': 'Grade 2 Hypertension',
        'HTN_gd3': 'Grade 3 Hypertension',
        'DIAB_HST': 'History of Diabetes',
        'DM_poorly_controlled': 'Poorly Controlled Diabetes',
        'Prediabetes': 'Prediabetes',
        'HYPERLIPIDEMIA_HST': 'History of Hyperlipidemia',
        'CAD_HST': 'History of Coronary Artery Disease',
        'CCI_MI': 'Myocardial Infarction',
        'REVASC_HST': 'History of Revascularization',
        'PRE_CABG': 'Prior Coronary Artery Bypass Grafting',
        'CCI_PERIPHERAL_VASC': 'Peripheral Vascular Disease',
        'ALL_HF': 'History of Heart Failure',
        'HFrEF': 'Heart Failure with Reduced Ejection Fraction',
        'HFmrEF': 'Heart Failure with Mid-Range Ejection Fraction',
        'HFpEF': 'Heart Failure with Preserved Ejection Fraction',
        'PAROXYSMAL_AF': 'Paroxysmal Atrial Fibrillation',
        'LONGSTNDING_AF': 'Long-standing Atrial Fibrillation',
        'OTHER_AF': 'Persistent Atrial Fibrillation',
        'PRE_AV_NODE_ABLATION': 'Prior AV Node Ablation',
        'PRE_PVI_ABLATION': 'Prior Pulmonary Vein Isolation Ablation',
        'ABLATION_EVER': 'History of Ablation',
        'PRE_CARDIOVERSION': 'Prior Cardioversion',
        'ELIX_CARDIAC_ARRTHYTHMIAS': 'Cardiac Arrhythmias',
        'VTVF_HST': 'History of Ventricular Tachycardia or Ventricular Fibrillation',
        'CARDIAC_ARREST_HST': 'History of Cardiac Arrest',
        'PRE_PPM_IMPLANT': 'Prior Permanent Pacemaker Implantation',
        'PRE_ICD_IMPLANT': 'Prior Implantable Cardioverter-Defibrillator Implantation',
        'PRE_CRT_D_IMPLANT': 'Prior Cardiac Resynchronization Therapy Defibrillator Implantation',
        'PRE_CRT_P_IMPLANT': 'Prior Cardiac Resynchronization Therapy Pacemaker Implantation',
        'ELIX_VALVULAR_DISEASE': 'Valvular Disease',
        'PRE_SEVERE_MV_REGURG': 'Prior Severe Mitral Valve Regurgitation',
        'PRE_SEVERE_AV_STEN': 'Prior Severe Aortic Valve Stenosis',
        'HYPERTROPHIC_CMP_HX': 'History of Hypertrophic Cardiomyopathy',
        'CCI_RENAL_DISEASE': 'Renal Disease',
        'CKD_2': 'Chronic Kidney Disease Stage 2',
        'CKD_3': 'Chronic Kidney Disease Stage 3',
        'CKD_4': 'Chronic Kidney Disease Stage 4',
        'CKD_5': 'Chronic Kidney Disease Stage 5',
        'ELIX_FLUID_ELECTROLYTE_DIS': 'Fluid and Electrolyte Disorders',
        'STROKE_HST': 'History of Stroke',
        'HEMO_STROKE_HST': 'History of Hemorrhagic Stroke',
        'TIA_HST': 'History of Transient Ischemic Attack',
        'ELIX_PARALYSIS': 'Paralysis',
        'ACTIVE_CANCER': 'Active Cancer',
        'LUNG_CANCER_HST': 'History of Lung Cancer',
        'ELIX_LYMPHOMA': 'Lymphoma history',
        'CCI_COPD': 'Chronic Obstructive Pulmonary Disease',
        'PH_HST': 'History of Pulmonary Hypertension',
        'ASTHMA_HST': 'History of Asthma',
        'OBS_SLEEPAPNEA_HST': 'History of Obstructive Sleep Apnea',
        'ELIX_PULM_CIRC_DISORDERS': 'Pulmonary Circulation Disorders',
        'PE_HST': 'History of Pulmonary Embolism',
        'DVT_HST': 'History of Deep Venous Thrombosis',
        'MAJOR_BLEED_HST': 'History of Major Bleeding',
        'ELIX_COAG_DEFICIENCY': 'Coagulation Deficiency',
        'Anemia_history': 'History of Anemia',
        'CCI_PEPTIC_ULCER': 'Peptic Ulcer Disease',
        'CCI_DEMENTIA': 'Dementia',
        'ELIX_PSYCHOSES': 'Psychoses',
        'ELIX_DEPRESSION': 'Depression',
        'CCI_RHEUMATIC_DISEASE': 'Rheumatoid Disease',
        'AMYLOID_HX': 'History of Amyloidosis',
        'SARCOIDOSIS_HX': 'History of Sarcoidosis',
        'CCI_AIDS_HIV': 'AIDS/HIV',
        'ELIX_HYPOTHYROIDISM': 'Hypothyroidism',
        'ELIX_LIVER_DISEASE': 'Liver Disease',
        'HASBLED_score': 'HASBLED Score',
        'ATRIA_score': 'ATRIA Score',
        'ORBIT_score': 'ORBIT Score',
        'CHA2DS2_VASc_score': 'CHA2DS2-VASc Score',
        'ASA': 'Aspirin',
        'BB': 'Beta Blockers',
        'CCB': 'Calcium Channel Blockers',
        'MRA': 'Mineralocorticoid Receptor Antagonists',
        'ACEI': 'Angiotensin-Converting Enzyme Inhibitors',
        'ARB': 'Angiotensin II Receptor Blockers',
        'ARNI': 'Angiotensin Receptor-Neprilysin Inhibitors',
        'STATIN': 'Statins',
        'SGLT2_INHIBITORS': 'SGLT2 Inhibitors',
        'INSULIN': 'Insulin',
        'METFORMIN': 'Metformin',
        'DPP4_INHIBTOR': 'DPP-4 Inhibitors',
        'TZD': 'Thiazolidinediones',
        'GLP1': 'GLP-1 Receptor Agonists',
        'SULFONYLUREA': 'Sulfonylureas',
        'RHYTHM_EVER': 'History of Any Rhythm Control Medications',
        'CLASS_I': 'Class I Antiarrhythmic Agents',
        'CLASS_III': 'Class III Antiarrhythmic Agents',
        'AMIODARONE': 'Amiodarone',
        'SOTALOL': 'Sotalol',
        'DOFETILIDE': 'Dofetilide',
        'FLECAINIDE': 'Flecainide',
        'PROPAFENONE': 'Propafenone',
        'CHOLESTEROL': 'Total Cholesterol (milligrams per deciliter)',
        'LDL': 'Low-Density Lipoprotein Cholesterol (milligrams per deciliter)',
        'HDL': 'High-Density Lipoprotein Cholesterol (milligrams per deciliter)',
        'TRIGLYCERIDE': 'Triglycerides (milligrams per deciliter)',
        'EGFR': 'Estimated Glomerular Filtration Rate (milliliters per minute per 1.73 square meters)',
        'CR': 'Creatinine (milligrams per deciliter)',
        'TSH': 'Thyroid Stimulating Hormone (micro international units per milliliter)',
        'LVEF_VISIT': 'Left Ventricular Ejection Fraction on prior transthoracic echocardiography (%)',
        'LA_DIA_VISIT': 'Left Atrial Diameter on prior transthoracic echocardiography (millimeters)',
        'LVDD_VISIT': 'Left Ventricular Diastolic Diameter on prior transthoracic echocardiography (millimeters)',
        'LVSD_VISIT': 'Left Ventricular Systolic Diameter on prior transthoracic echocardiography (millimeters)',
        'MR_MODSEVERE_VISIT': 'Moderate to Severe Mitral Regurgitation on prior transthoracic echocardiography',
        'MR_SEVERE_VISIT': 'Severe Mitral Regurgitation on prior transthoracic echocardiography',
        'MS_MODSEVERE_VISIT': 'Moderate to Severe Mitral Stenosis on prior transthoracic echocardiography',
        'MS_SEVERE_VISIT': 'Severe Mitral Stenosis on prior transthoracic echocardiography',
        'NONCARDIAC_READMIT_FLAG': 'Non-Cardiac Readmission',
        'CARDIAC_READMIT_FLAG': 'Cardiac Readmission',
        'STEMI_READMIT_FLAG': 'ST-Elevation Myocardial Infarction Readmission',
        'NSTEMI_REVASC_READMIT_FLAG': 'Non-ST-Elevation Myocardial Infarction Revascularization Readmission',
        'USA_REVASC_READMIT_FLAG': 'Unstable Angina Revascularization Readmission',
        'SA_REVASC_READMIT_FLAG': 'Stable Angina Revascularization Readmission',
        'I_STROKE_READMIT_FLAG': 'Ischemic Stroke Readmission',
        'H_STROKE_READMIT_FLAG': 'Hemorrhagic Stroke Readmission',
        'TIA_READMIT_FLAG': 'Transient Ischemic Attack Readmission',
        'MAJOR_BLEED_READMIT_FLAG': 'Major Bleeding Readmission',
        'CTB': 'All-Cause Death',
        'MBE_1year': 'Major Bleeding Event Within 1 Year',
        'MBE_1year_TIME': 'Time to Major Bleeding Event Within 1 Year (days)'
    }
    return name_mapping.get(variable_name, variable_name)

def generate_summary_table(df_filtered, categorical_cols, continuous_cols, outcome, output_path, mapping_order):
    summary_table = []
    
    overall_count = len(df_filtered)
    group1_count = len(df_filtered[df_filtered[outcome] == 1])
    group2_count = len(df_filtered[df_filtered[outcome] == 0])
    
    summary_table.append(['Number of Patients', f'Overall (n={overall_count})', f"{outcome}=1 (n={group1_count})", f"{outcome}=0 (n={group2_count})", '', ''])
    
    for column in mapping_order:
        if column not in df_filtered.columns or (column not in categorical_cols and column not in continuous_cols) or column in ['CCI_RENAL_DISEASE', 'MAJOR_BLEED_READMIT_FLAG']:
            continue
        
        is_continuous = column in continuous_cols
        
        overall_stats = summary_stats(df_filtered, column, outcome, is_continuous)
        group1_stats = summary_stats(df_filtered[df_filtered[outcome] == 1], column, outcome, is_continuous)
        
        if column == 'MBE_1year_TIME':
            overall_stats = '-'
            group2_stats = '-'
        else:
            overall_stats = summary_stats(df_filtered, column, outcome, is_continuous)
            group2_stats = summary_stats(df_filtered[df_filtered[outcome] == 0], column, outcome, is_continuous)
        
        p_val = compare_groups(df_filtered, column, outcome, is_continuous)
        extrapolated_name = extrapolate_name(column)
        
        summary_table.append([column, overall_stats, group1_stats, group2_stats, p_val, extrapolated_name])
    
    summary_df = pd.DataFrame(summary_table, columns=['Variable', 'Overall', f"{outcome}=1", f"{outcome}=0", 'P-value', 'Extrapolated Name'])
    summary_df.to_csv(output_path, index=False)

def generate_summary_tables(train_set, test_set_non_comorbid, test_set_random,
                            categorical_cols_train, continuous_cols_train,
                            categorical_cols_non_comorbid, continuous_cols_non_comorbid,
                            categorical_cols_random, continuous_cols_random,
                            summary_table_train_path, summary_table_test_non_comorbid_path, summary_table_test_random_path,
                            outcome, timepoint):
    outcome_var = f'{outcome}_{timepoint}'
    mapping_order = [
        'AGE',
        'AGE_LESS_THAN_50',
        'AGE_50TO64',
        'AGE_65TO74',
        'AGE_MORE_THAN_75',
        'FEMALE',
        'White_Race',
        'Black_Race',
        'Asian_Race',
        'Other_Races',
        'EthnicGroup',
        'EVER_SMOKER',
        'CURRENT_SMOKER',
        'EVER_ALCOHOL',
        'CURRENT_ALCOHOL',
        'EVER_ILL_DRUG',
        'CURRENT_ILL_DRUG',
        'INS_MC',
        'INS_MA',
        'BMI',
        'Obese',
        'Underweight',
        'Poorly_controlled_HTN_SBPmoreTHAN149',
        'Bradycardic',
        'Tachycardic',
        'ELIX_HYPERTENSION',
        'HTN_gd1',
        'HTN_gd2',
        'HTN_gd3',
        'DIAB_HST',
        'DM_poorly_controlled',
        'Prediabetes',
        'HYPERLIPIDEMIA_HST',
        'CAD_HST',
        'CCI_MI',
        'REVASC_HST',
        'PRE_CABG',
        'CCI_PERIPHERAL_VASC',
        'ALL_HF',
        'HFrEF',
        'HFmrEF',
        'HFpEF',
        'PAROXYSMAL_AF',
        'LONGSTNDING_AF',
        'OTHER_AF',
        'PRE_AV_NODE_ABLATION',
        'PRE_PVI_ABLATION',
        'ABLATION_EVER',
        'PRE_CARDIOVERSION',
        'ELIX_CARDIAC_ARRTHYTHMIAS',
        'VTVF_HST',
        'CARDIAC_ARREST_HST',
        'PRE_PPM_IMPLANT',
        'PRE_ICD_IMPLANT',
        'PRE_CRT_D_IMPLANT',
        'PRE_CRT_P_IMPLANT',
        'ELIX_VALVULAR_DISEASE',
        'PRE_SEVERE_MV_REGURG',
        'PRE_SEVERE_AV_STEN',
        'HYPERTROPHIC_CMP_HX',
        'CCI_RENAL_DISEASE',
        'CKD_2',
        'CKD_3',
        'CKD_4',
        'CKD_5',
        'ELIX_FLUID_ELECTROLYTE_DIS',
        'STROKE_HST',
        'HEMO_STROKE_HST',
        'TIA_HST',
        'ELIX_PARALYSIS',
        'ACTIVE_CANCER',
        'LUNG_CANCER_HST',
        'ELIX_LYMPHOMA',
        'CCI_COPD',
        'PH_HST',
        'ASTHMA_HST',
        'OBS_SLEEPAPNEA_HST',
        'ELIX_PULM_CIRC_DISORDERS',
        'PE_HST',
        'DVT_HST',
        'MAJOR_BLEED_HST',
        'ELIX_COAG_DEFICIENCY',
        'Anemia_history',
        'CCI_PEPTIC_ULCER',
        'CCI_DEMENTIA',
        'ELIX_PSYCHOSES',
        'ELIX_DEPRESSION',
        'CCI_RHEUMATIC_DISEASE',
        'AMYLOID_HX',
        'SARCOIDOSIS_HX',
        'CCI_AIDS_HIV',
        'ELIX_HYPOTHYROIDISM',
        'ELIX_LIVER_DISEASE',
        'HASBLED_score',
        'ATRIA_score',
        'ORBIT_score',
        'CHA2DS2_VASc_score',
        'ASA',
        'BB',
        'CCB',
        'MRA',
        'ACEI',
        'ARB',
        'ARNI',
        'STATIN',
        'SGLT2_INHIBITORS',
        'INSULIN',
        'METFORMIN',
        'DPP4_INHIBTOR',
        'TZD',
        'GLP1',
        'SULFONYLUREA',
        'RHYTHM_EVER',
        'CLASS_I',
        'CLASS_III',
        'AMIODARONE',
        'SOTALOL',
        'DOFETILIDE',
        'FLECAINIDE',
        'PROPAFENONE',
        'CHOLESTEROL',
        'LDL',
        'HDL',
        'TRIGLYCERIDE',
        'EGFR',
        'CR',
        'TSH',
        'LVEF_VISIT',
        'LA_DIA_VISIT',
        'LVDD_VISIT',
        'LVSD_VISIT',
        'MR_MODSEVERE_VISIT',
        'MR_SEVERE_VISIT',
        'MS_MODSEVERE_VISIT',
        'MS_SEVERE_VISIT',
        'NONCARDIAC_READMIT_FLAG',
        'CARDIAC_READMIT_FLAG',
        'STEMI_READMIT_FLAG',
        'NSTEMI_REVASC_READMIT_FLAG',
        'USA_REVASC_READMIT_FLAG',
        'SA_REVASC_READMIT_FLAG',
        'I_STROKE_READMIT_FLAG',
        'H_STROKE_READMIT_FLAG',
        'TIA_READMIT_FLAG',
        'MAJOR_BLEED_READMIT_FLAG',
        'CTB',
        'MBE_1year',
        'MBE_1year_TIME'
    ]
    generate_summary_table(train_set, categorical_cols_train, continuous_cols_train, outcome_var, summary_table_train_path, mapping_order)
    generate_summary_table(test_set_non_comorbid, categorical_cols_non_comorbid, continuous_cols_non_comorbid, outcome_var, summary_table_test_non_comorbid_path, mapping_order)
    generate_summary_table(test_set_random, categorical_cols_random, continuous_cols_random, outcome_var, summary_table_test_random_path, mapping_order)


# block7 code
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

def extrapolate_name(variable_name):
    name_mapping = {
        'AGE': 'Age (Years)',
        'AGE_LESS_THAN_50': 'Age Less Than 50 Years',
        'AGE_50TO64': 'Age 50 to 64 Years',
        'AGE_65TO74': 'Age 65 to 74 Years',
        'AGE_MORE_THAN_75': 'Age More Than 75 Years',
        'FEMALE': 'Sex (Female)',
        'White_Race': 'White Race',
        'Black_Race': 'Black Race',
        'Asian_Race': 'Asian Race',
        'Other_Races': 'Other Races',
        'EthnicGroup': 'Hispanic or Latino Ethnic Group',
        'EVER_SMOKER': 'Ever Smoker',
        'CURRENT_SMOKER': 'Current Smoker',
        'EVER_ALCOHOL': 'Ever Alcohol Use',
        'CURRENT_ALCOHOL': 'Current Alcohol Use',
        'EVER_ILL_DRUG': 'Ever Illicit Drug Use',
        'CURRENT_ILL_DRUG': 'Current Illicit Drug Use',
        'INS_MC': 'Medicare Insurance',
        'INS_MA': 'Medicaid Insurance',
        'BMI': 'Body Mass Index (kilograms per square meter)',
        'Obese': 'Obese with BMI >29.9',
        'Underweight': 'Underweight with BMI <18.5',
        'Poorly_controlled_HTN_SBPmoreTHAN149': 'Poorly Controlled Hypertension (Systolic Blood Pressure More Than 149 mmHg)',
        'Bradycardic': 'Bradycardic with heart rate <60 beats per minute',
        'Tachycardic': 'Tachycardic with heart rate >100 beats per minute',
        'ELIX_HYPERTENSION': 'Hypertension',
        'HTN_gd1': 'Grade 1 Hypertension',
        'HTN_gd2': 'Grade 2 Hypertension',
        'HTN_gd3': 'Grade 3 Hypertension',
        'DIAB_HST': 'History of Diabetes',
        'DM_poorly_controlled': 'Poorly Controlled Diabetes',
        'Prediabetes': 'Prediabetes',
        'HYPERLIPIDEMIA_HST': 'History of Hyperlipidemia',
        'CAD_HST': 'History of Coronary Artery Disease',
        'CCI_MI': 'Myocardial Infarction',
        'REVASC_HST': 'History of Revascularization',
        'PRE_CABG': 'Prior Coronary Artery Bypass Grafting',
        'CCI_PERIPHERAL_VASC': 'Peripheral Vascular Disease',
        'ALL_HF': 'History of Heart Failure',
        'HFrEF': 'Heart Failure with Reduced Ejection Fraction',
        'HFmrEF': 'Heart Failure with Mid-Range Ejection Fraction',
        'HFpEF': 'Heart Failure with Preserved Ejection Fraction',
        'PAROXYSMAL_AF': 'Paroxysmal Atrial Fibrillation',
        'LONGSTNDING_AF': 'Long-standing Atrial Fibrillation',
        'OTHER_AF': 'Persistent Atrial Fibrillation',
        'PRE_AV_NODE_ABLATION': 'Prior AV Node Ablation',
        'PRE_PVI_ABLATION': 'Prior Pulmonary Vein Isolation Ablation',
        'ABLATION_EVER': 'History of Ablation',
        'PRE_CARDIOVERSION': 'Prior Cardioversion',
        'ELIX_CARDIAC_ARRTHYTHMIAS': 'Cardiac Arrhythmias',
        'VTVF_HST': 'History of Ventricular Tachycardia or Ventricular Fibrillation',
        'CARDIAC_ARREST_HST': 'History of Cardiac Arrest',
        'PRE_PPM_IMPLANT': 'Prior Permanent Pacemaker Implantation',
        'PRE_ICD_IMPLANT': 'Prior Implantable Cardioverter-Defibrillator Implantation',
        'PRE_CRT_D_IMPLANT': 'Prior Cardiac Resynchronization Therapy Defibrillator Implantation',
        'PRE_CRT_P_IMPLANT': 'Prior Cardiac Resynchronization Therapy Pacemaker Implantation',
        'ELIX_VALVULAR_DISEASE': 'Valvular Disease',
        'PRE_SEVERE_MV_REGURG': 'Prior Severe Mitral Valve Regurgitation',
        'PRE_SEVERE_AV_STEN': 'Prior Severe Aortic Valve Stenosis',
        'HYPERTROPHIC_CMP_HX': 'History of Hypertrophic Cardiomyopathy',
        'CKD_2': 'Chronic Kidney Disease Stage 2',
        'CKD_3': 'Chronic Kidney Disease Stage 3',
        'CKD_4': 'Chronic Kidney Disease Stage 4',
        'CKD_5': 'Chronic Kidney Disease Stage 5',
        'ELIX_FLUID_ELECTROLYTE_DIS': 'Fluid and Electrolyte Disorders',
        'STROKE_HST': 'History of Stroke',
        'HEMO_STROKE_HST': 'History of Hemorrhagic Stroke',
        'TIA_HST': 'History of Transient Ischemic Attack',
        'ELIX_PARALYSIS': 'Paralysis',
        'ACTIVE_CANCER': 'Active Cancer',
        'LUNG_CANCER_HST': 'History of Lung Cancer',
        'ELIX_LYMPHOMA': 'Lymphoma history',
        'CCI_COPD': 'Chronic Obstructive Pulmonary Disease',
        'PH_HST': 'History of Pulmonary Hypertension',
        'ASTHMA_HST': 'History of Asthma',
        'OBS_SLEEPAPNEA_HST': 'History of Obstructive Sleep Apnea',
        'ELIX_PULM_CIRC_DISORDERS': 'Pulmonary Circulation Disorders',
        'PE_HST': 'History of Pulmonary Embolism',
        'DVT_HST': 'History of Deep Venous Thrombosis',
        'MAJOR_BLEED_HST': 'History of Major Bleeding',
        'ELIX_COAG_DEFICIENCY': 'Coagulation Deficiency',
        'Anemia_history': 'History of Anemia',
        'CCI_PEPTIC_ULCER': 'Peptic Ulcer Disease',
        'CCI_DEMENTIA': 'Dementia',
        'ELIX_PSYCHOSES': 'Psychoses',
        'ELIX_DEPRESSION': 'Depression',
        'CCI_RHEUMATIC_DISEASE': 'Rheumatoid Disease',
        'AMYLOID_HX': 'History of Amyloidosis',
        'SARCOIDOSIS_HX': 'History of Sarcoidosis',
        'CCI_AIDS_HIV': 'AIDS/HIV',
        'ELIX_HYPOTHYROIDISM': 'Hypothyroidism',
        'ELIX_LIVER_DISEASE': 'Liver Disease',
        'HASBLED_score': 'HASBLED Score',
        'ATRIA_score': 'ATRIA Score',
        'ORBIT_score': 'ORBIT Score',
        'CHA2DS2_VASc_score': 'CHA2DS2-VASc Score',
        'ASA': 'Aspirin',
        'BB': 'Beta Blockers',
        'CCB': 'Calcium Channel Blockers',
        'MRA': 'Mineralocorticoid Receptor Antagonists',
        'ACEI': 'Angiotensin-Converting Enzyme Inhibitors',
        'ARB': 'Angiotensin II Receptor Blockers',
        'ARNI': 'Angiotensin Receptor-Neprilysin Inhibitors',
        'STATIN': 'Statins',
        'SGLT2_INHIBITORS': 'SGLT2 Inhibitors',
        'INSULIN': 'Insulin',
        'METFORMIN': 'Metformin',
        'DPP4_INHIBTOR': 'DPP-4 Inhibitors',
        'TZD': 'Thiazolidinediones',
        'GLP1': 'GLP-1 Receptor Agonists',
        'SULFONYLUREA': 'Sulfonylureas',
        'RHYTHM_EVER': 'History of Any Rhythm Control Medications',
        'CLASS_I': 'Class I Antiarrhythmic Agents',
        'CLASS_III': 'Class III Antiarrhythmic Agents',
        'AMIODARONE': 'Amiodarone',
        'SOTALOL': 'Sotalol',
        'DOFETILIDE': 'Dofetilide',
        'FLECAINIDE': 'Flecainide',
        'PROPAFENONE': 'Propafenone',
        'CHOLESTEROL': 'Total Cholesterol (milligrams per deciliter)',
        'LDL': 'Low-Density Lipoprotein Cholesterol (milligrams per deciliter)',
        'HDL': 'High-Density Lipoprotein Cholesterol (milligrams per deciliter)',
        'TRIGLYCERIDE': 'Triglycerides (milligrams per deciliter)',
        'EGFR': 'Estimated Glomerular Filtration Rate (milliliters per minute per 1.73 square meters)',
        'CR': 'Creatinine (milligrams per deciliter)',
        'TSH': 'Thyroid Stimulating Hormone (micro international units per milliliter)',
        'LVEF_VISIT': 'Left Ventricular Ejection Fraction on prior transthoracic echocardiography (%)',
        'LA_DIA_VISIT': 'Left Atrial Diameter on prior transthoracic echocardiography (millimeters)',
        'LVDD_VISIT': 'Left Ventricular Diastolic Diameter on prior transthoracic echocardiography (millimeters)',
        'LVSD_VISIT': 'Left Ventricular Systolic Diameter on prior transthoracic echocardiography (millimeters)',
        'MR_MODSEVERE_VISIT': 'Moderate to Severe Mitral Regurgitation on prior transthoracic echocardiography',
        'MR_SEVERE_VISIT': 'Severe Mitral Regurgitation on prior transthoracic echocardiography',
        'MS_MODSEVERE_VISIT': 'Moderate to Severe Mitral Stenosis on prior transthoracic echocardiography',
        'MS_SEVERE_VISIT': 'Severe Mitral Stenosis on prior transthoracic echocardiography',
        'NONCARDIAC_READMIT_FLAG': 'Non-Cardiac Readmission',
        'CARDIAC_READMIT_FLAG': 'Cardiac Readmission',
        'STEMI_READMIT_FLAG': 'ST-Elevation Myocardial Infarction Readmission',
        'NSTEMI_REVASC_READMIT_FLAG': 'Non-ST-Elevation Myocardial Infarction Revascularization Readmission',
        'USA_REVASC_READMIT_FLAG': 'Unstable Angina Revascularization Readmission',
        'SA_REVASC_READMIT_FLAG': 'Stable Angina Revascularization Readmission',
        'I_STROKE_READMIT_FLAG': 'Ischemic Stroke Readmission',
        'H_STROKE_READMIT_FLAG': 'Hemorrhagic Stroke Readmission',
        'TIA_READMIT_FLAG': 'Transient Ischemic Attack Readmission',
        'CTB': 'All-Cause Death',
        'MBE_1year': 'Major Bleeding Event Within 1 Year',
        'MBE_1year_TIME': 'Time to Major Bleeding Event Within 1 Year (days)'
    }
    return name_mapping.get(variable_name, variable_name)
def calculate_missing_values(df_filtered):
    missing_values = []
    for col in df_filtered.columns:
        pct_missing = np.mean(df_filtered[col].isnull()) * 100
        extrapolated_name = extrapolate_name(col)
        missing_values.append([col, extrapolated_name, round(pct_missing, 1)])
    
    missing_values_df = pd.DataFrame(missing_values, columns=['Variable', 'Extrapolated Name', 'Percentage Missing'])
    missing_values_df = missing_values_df.set_index('Variable')
    
    return missing_values_df

def generate_vif_correlation(df_filtered, categorical_cols_filtered, continuous_cols_filtered, corr_threshold=0.8):
    # Create a temporary copy of the training set for imputation
    df_imputed = df_filtered.copy()

    # Check for missing values in categorical columns and impute with most frequent value
    for col in categorical_cols_filtered:
        if df_imputed[col].isnull().any():
            most_frequent = df_imputed[col].mode()[0]
            df_imputed[col] = df_imputed[col].fillna(most_frequent)

    # Check for missing values in continuous columns and impute with median value
    for col in continuous_cols_filtered:
        if df_imputed[col].isnull().any():
            median = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median)

    # Ensure the data for VIF calculation only includes the desired continuous columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_imputed[continuous_cols_filtered])

    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data['Variable'] = continuous_cols_filtered
    vif_data['Extrapolated Name'] = vif_data['Variable'].apply(extrapolate_name)
    vif_data['VIF'] = [variance_inflation_factor(scaled_data, i) for i in range(scaled_data.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False)

    # Calculate the correlation matrix for categorical variables (if they are numerically encoded)
    corr_matrix = df_imputed[categorical_cols_filtered].corr().abs()

    # Filter the matrix for high correlations based on the specified threshold
    high_corr_var = np.where(corr_matrix > corr_threshold)
    high_corr_var = [(extrapolate_name(corr_matrix.columns[x]), extrapolate_name(corr_matrix.columns[y])) for x, y in zip(*high_corr_var) if x != y and x < y]

    return vif_data, high_corr_var

def generate_missing_values_and_vif_correlation(train_set, test_set_non_comorbid, test_set_random,
                                                categorical_cols_train, continuous_cols_train,
                                                missing_values_train_path, missing_values_test_non_comorbid_path,
                                                missing_values_test_random_path, vif_correlation_train_path,
                                                high_corr_categorical_train_path, corr_threshold=0.8):
    # Calculate missing values for the training set
    missing_values_train = calculate_missing_values(train_set)
    missing_values_train.to_csv(missing_values_train_path, index=True)

    # Calculate missing values for the non-comorbid test set
    missing_values_test_non_comorbid = calculate_missing_values(test_set_non_comorbid)
    missing_values_test_non_comorbid.to_csv(missing_values_test_non_comorbid_path, index=True)

    # Calculate missing values for the random test set
    missing_values_test_random = calculate_missing_values(test_set_random)
    missing_values_test_random.to_csv(missing_values_test_random_path, index=True)

    # Calculate VIF and correlations only for the training set
    vif_data, high_corr_var = generate_vif_correlation(train_set, categorical_cols_train, continuous_cols_train, corr_threshold)
    vif_data.to_csv(vif_correlation_train_path, index=False)

    # Save high correlation values for categorical variables to a CSV file
    high_corr_categorical_df = pd.DataFrame(high_corr_var, columns=['Variable 1', 'Variable 2'])
    high_corr_categorical_df.to_csv(high_corr_categorical_train_path, index=False)

# block8 code
import pandas as pd
from sklearn.impute import SimpleImputer

def drop_high_missing_columns(train_set, test_set_non_comorbid, test_set_random, missing_threshold=60):
    # Calculate the percentage of missing values for each column in the training set
    missing_percentages = train_set.isnull().mean() * 100
    
    # Identify columns with missing values above the threshold
    columns_to_drop = missing_percentages[missing_percentages > missing_threshold].index.tolist()
    
    # Drop the identified columns from all datasets
    train_set = train_set.drop(columns=columns_to_drop)
    test_set_non_comorbid = test_set_non_comorbid.drop(columns=columns_to_drop)
    test_set_random = test_set_random.drop(columns=columns_to_drop)
    
    return train_set, test_set_non_comorbid, test_set_random, columns_to_drop

def drop_highly_correlated_columns(train_set, test_set_non_comorbid, test_set_random, high_corr_categorical_train_path):
    # Read the CSV file containing highly correlated categorical variables
    high_corr_categorical_df = pd.read_csv(high_corr_categorical_train_path)
    
    # Identify columns to drop based on high correlation
    columns_to_drop = high_corr_categorical_df['Variable 1'].tolist()
    
    # Drop the identified columns from all datasets if they exist
    train_set = train_set.drop(columns=columns_to_drop, errors='ignore')
    test_set_non_comorbid = test_set_non_comorbid.drop(columns=columns_to_drop, errors='ignore')
    test_set_random = test_set_random.drop(columns=columns_to_drop, errors='ignore')
    
    # Update columns_to_drop to only include columns that were actually dropped
    columns_to_drop = [col for col in columns_to_drop if col in train_set.columns]
    
    return train_set, test_set_non_comorbid, test_set_random, columns_to_drop

def drop_specific_columns(train_set, test_set_non_comorbid, test_set_random):
    # Define the specific columns to drop
    columns_to_drop = [
        'AGE', 'ORBIT_score', 'ATRIA_score', 'CHA2DS2_VASc_score', 'HASBLED_score', 'MBE_1year_TIME',
        's.no', 'DIAB_HST', 'EGFR', 'ALL_HF', 'NONCARDIAC_READMIT_FLAG', 'CARDIAC_READMIT_FLAG',
        'STEMI_READMIT_FLAG', 'NSTEMI_REVASC_READMIT_FLAG', 'USA_REVASC_READMIT_FLAG',
        'SA_REVASC_READMIT_FLAG', 'I_STROKE_READMIT_FLAG', 'H_STROKE_READMIT_FLAG',
        'TIA_READMIT_FLAG', 'MAJOR_BLEED_READMIT_FLAG', 'CTB', 'EGFR', 'BMI', 'LVEF_VISIT', 'CR', 'CHOLESTEROL', 'LDL', 'HDL', 'TRIGLYCERIDE','TSH'

    ]
    
    # Drop the specific columns from all datasets
    train_set = train_set.drop(columns=columns_to_drop, errors='ignore')
    test_set_non_comorbid = test_set_non_comorbid.drop(columns=columns_to_drop, errors='ignore')
    test_set_random = test_set_random.drop(columns=columns_to_drop, errors='ignore')
    
    return train_set, test_set_non_comorbid, test_set_random, columns_to_drop

def impute_missing_values(df_filtered, categorical_cols, continuous_cols):
    # Create separate DataFrames for categorical and continuous columns
    df_filtered_categorical = df_filtered[categorical_cols]
    
    # Impute missing values in categorical columns using most frequent value
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df_filtered_categorical_imputed = pd.DataFrame(imputer_categorical.fit_transform(df_filtered_categorical), columns=categorical_cols, index=df_filtered.index)
    
    if len(continuous_cols) > 0:
        df_filtered_continuous = df_filtered[continuous_cols]
        
        # Impute missing values in continuous columns using median
        imputer_continuous = SimpleImputer(strategy='median')
        df_filtered_continuous_imputed = pd.DataFrame(imputer_continuous.fit_transform(df_filtered_continuous), columns=continuous_cols, index=df_filtered.index)
        
        # Combine the imputed DataFrames
        df_filtered_imputed = pd.concat([df_filtered_categorical_imputed, df_filtered_continuous_imputed], axis=1)
    else:
        df_filtered_imputed = df_filtered_categorical_imputed
    
    return df_filtered_imputed

def preprocess_datasets(train_set, test_set_non_comorbid, test_set_random, categorical_cols_train, continuous_cols_train, categorical_cols_non_comorbid, continuous_cols_non_comorbid, categorical_cols_random, continuous_cols_random, high_corr_categorical_train_path, preprocessed_train_set_path, preprocessed_test_set_non_comorbid_path, preprocessed_test_set_random_path):
    # Drop columns with high missing values
    train_set, test_set_non_comorbid, test_set_random, columns_dropped_missing = drop_high_missing_columns(train_set, test_set_non_comorbid, test_set_random)
    
    # Drop highly correlated columns
    train_set, test_set_non_comorbid, test_set_random, columns_dropped_corr = drop_highly_correlated_columns(train_set, test_set_non_comorbid, test_set_random, high_corr_categorical_train_path)
    
    # Drop specific columns
    train_set, test_set_non_comorbid, test_set_random, columns_dropped_specific = drop_specific_columns(train_set, test_set_non_comorbid, test_set_random)
    
    # Update categorical and continuous column lists for each dataset
    categorical_cols_train = [col for col in categorical_cols_train if col not in columns_dropped_missing + columns_dropped_corr + columns_dropped_specific]
    continuous_cols_train = [col for col in continuous_cols_train if col not in columns_dropped_missing + columns_dropped_corr + columns_dropped_specific]
    categorical_cols_non_comorbid = [col for col in categorical_cols_non_comorbid if col not in columns_dropped_missing + columns_dropped_corr + columns_dropped_specific]
    continuous_cols_non_comorbid = [col for col in continuous_cols_non_comorbid if col not in columns_dropped_missing + columns_dropped_corr + columns_dropped_specific]
    categorical_cols_random = [col for col in categorical_cols_random if col not in columns_dropped_missing + columns_dropped_corr + columns_dropped_specific]
    continuous_cols_random = [col for col in continuous_cols_random if col not in columns_dropped_missing + columns_dropped_corr + columns_dropped_specific]
    
    # Impute missing values in each dataset separately
    train_set_imputed = impute_missing_values(train_set, categorical_cols_train, continuous_cols_train)
    test_set_non_comorbid_imputed = impute_missing_values(test_set_non_comorbid, categorical_cols_non_comorbid, continuous_cols_non_comorbid)
    test_set_random_imputed = impute_missing_values(test_set_random, categorical_cols_random, continuous_cols_random)

    # Save the imputed datasets to CSV files
    train_set_imputed.to_csv(preprocessed_train_set_path, index=False)
    test_set_non_comorbid_imputed.to_csv(preprocessed_test_set_non_comorbid_path, index=False)
    test_set_random_imputed.to_csv(preprocessed_test_set_random_path, index=False)
    
    return train_set_imputed, test_set_non_comorbid_imputed, test_set_random_imputed, categorical_cols_train, continuous_cols_train, categorical_cols_non_comorbid, continuous_cols_non_comorbid, categorical_cols_random, continuous_cols_random

#Block9 code
import pandas as pd

from imblearn.over_sampling import SMOTE

MY_RANDOM_STATE = 42

def perform_smote(train_set, categorical_cols, outcome_var, smote_train_set_path, smote_train_set_missing_path):
    # Convert boolean columns to numeric
    bool_cols = train_set.select_dtypes(include='bool').columns
    train_set[bool_cols] = train_set[bool_cols].astype(int)

    # Prepare data for imputation and SMOTE
    if outcome_var in train_set.columns:
        X_train = train_set.drop(columns=[outcome_var], errors='ignore')
        y_train = train_set[outcome_var]
    else:
        X_train = train_set
        y_train = None

    # Verify integrity before SMOTE
    if y_train is not None:
        print("Before SMOTE, counts of label '1':", sum(y_train == 1))
        print("Before SMOTE, counts of label '0':", sum(y_train == 0))
        print("Total missing values in X_train before SMOTE:", X_train.isnull().sum().sum())

    # Encode categorical variables using one-hot encoding
    existing_categorical_cols = [col for col in categorical_cols if col in X_train.columns]
    X_train_encoded = pd.get_dummies(X_train, columns=existing_categorical_cols)

    # Convert data types before applying SMOTE
    X_train_encoded = X_train_encoded.astype(float)
    y_train = y_train.astype(int)

    # Apply SMOTE
    if y_train is not None:
        smote = SMOTE(random_state=MY_RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)
        # Verify after SMOTE
        print("After SMOTE, counts of label '1':", sum(y_train_resampled == 1))
        print("After SMOTE, counts of label '0':", sum(y_train_resampled == 0))
    else:
        X_train_resampled = X_train_encoded
        y_train_resampled = None

    print("Total missing values in X_train_resampled after SMOTE:", X_train_resampled.isnull().sum().sum())

    # Combine the resampled features and target variable into a single DataFrame
    if y_train_resampled is not None:
        train_set_resampled = pd.concat([X_train_resampled, pd.Series(y_train_resampled, name=outcome_var)], axis=1)
    else:
        train_set_resampled = X_train_resampled

    # Save the SMOTE-enhanced training set to a CSV file
    train_set_resampled.to_csv(smote_train_set_path, index=False)

    # Calculate the missing percentage for each column in the SMOTE-enhanced training set
    missing_percentages = train_set_resampled.isnull().mean() * 100

    # Create a DataFrame with the missing percentages
    missing_percentages_df = pd.DataFrame({'Column': missing_percentages.index, 'Missing Percentage': missing_percentages.values})

    # Save the missing percentages to a CSV file
    missing_percentages_df.to_csv(smote_train_set_missing_path, index=False)

    return train_set_resampled

# Block10 code
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_encode_features(train_set_resampled, test_set_non_comorbid, test_set_random, categorical_cols, continuous_cols, outcome_var, scaled_train_set_path, scaled_test_set_non_comorbid_path, scaled_test_set_random_path):
    # Separate features and target variable for the training set
    X_train = train_set_resampled.drop(columns=[outcome_var], errors='ignore') if outcome_var in train_set_resampled.columns else train_set_resampled.copy()
    y_train = train_set_resampled[outcome_var] if outcome_var in train_set_resampled.columns else None

    # Apply one-hot encoding to the training set for categorical variables present
    categorical_cols_for_encoding = [col for col in categorical_cols if col in X_train.columns and col != outcome_var]
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols_for_encoding, drop_first=False)
    
    # Convert boolean columns to integers (excluding the outcome variable)
    bool_cols = X_train_encoded.select_dtypes(include='bool').columns
    X_train_encoded[bool_cols] = X_train_encoded[bool_cols].astype(int)
    
    # If y_train is not None, concatenate it back to form the complete training set
    if y_train is not None:
        train_set_scaled_encoded = pd.concat([X_train_encoded, y_train], axis=1)
    else:
        train_set_scaled_encoded = X_train_encoded

    # Save the processed training set
    train_set_scaled_encoded.to_csv(scaled_train_set_path, index=False)
    
    # Function to process test datasets
    def process_test_set(test_set, train_columns, categorical_cols):
        X_test = test_set.drop(columns=[outcome_var], errors='ignore') if outcome_var in test_set.columns else test_set.copy()
        y_test = test_set[outcome_var] if outcome_var in test_set.columns else None

        # Apply one-hot encoding to categorical variables
        categorical_cols_for_encoding = [col for col in categorical_cols if col in X_test.columns and col != outcome_var]
        X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols_for_encoding, drop_first=False)

        # Convert boolean columns to integers (excluding the outcome variable)
        bool_cols_test = X_test_encoded.select_dtypes(include='bool').columns
        X_test_encoded[bool_cols_test] = X_test_encoded[bool_cols_test].astype(int)
        
        # Add missing columns from the training set to the test set
        missing_cols = list(set(train_columns) - set(X_test_encoded.columns))
        missing_cols_df = pd.DataFrame(0, index=X_test_encoded.index, columns=missing_cols)
        X_test_encoded = pd.concat([X_test_encoded, missing_cols_df], axis=1)
        
        if y_test is not None:
            test_set_scaled_encoded = pd.concat([X_test_encoded, y_test], axis=1)
        else:
            test_set_scaled_encoded = X_test_encoded

        # Reorder columns to match the training set
        test_set_scaled_encoded = test_set_scaled_encoded[train_columns]

        return test_set_scaled_encoded
    
    # Process the test sets
    train_columns = train_set_scaled_encoded.columns.tolist()  # Columns of the training set after encoding and scaling
    test_set_non_comorbid_scaled_encoded = process_test_set(test_set_non_comorbid, train_columns, categorical_cols)
    test_set_random_scaled_encoded = process_test_set(test_set_random, train_columns, categorical_cols)
    
    # Save the processed test sets
    test_set_non_comorbid_scaled_encoded.to_csv(scaled_test_set_non_comorbid_path, index=False)
    test_set_random_scaled_encoded.to_csv(scaled_test_set_random_path, index=False)
    
    return train_set_scaled_encoded, test_set_non_comorbid_scaled_encoded, test_set_random_scaled_encoded

#block11
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import numpy as np

def initialize_models(models_dict):
   return {name: model() for name, model in models_dict.items()}

def perform_cross_validation(models, X_train, y_train, n_splits=5):
   skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
   cv_results = {name: [] for name in models}

   for train_index, val_index in skf.split(X_train, y_train):
       X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
       y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

       scaler = StandardScaler()
       X_train_fold_scaled = scaler.fit_transform(X_train_fold)
       X_val_fold_scaled = scaler.transform(X_val_fold)

       for name, model in models.items():
           model.fit(X_train_fold_scaled, y_train_fold)
           y_pred_proba = model.predict_proba(X_val_fold_scaled)[:, 1]
           cv_results[name].append(roc_auc_score(y_val_fold, y_pred_proba))

   return {name: sum(scores) / len(scores) for name, scores in cv_results.items()}

def calculate_performance_metrics(y_true, y_pred, y_pred_proba=None):
   metrics = {
       'Accuracy': accuracy_score(y_true, y_pred),
       'Precision': precision_score(y_true, y_pred, zero_division=0),
       'Recall': recall_score(y_true, y_pred),
       'F1 Score': f1_score(y_true, y_pred)
   }

   if y_pred_proba is not None:
       metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
       precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
       metrics['AUPRC'] = auc(recall, precision)

   return metrics

def evaluate_models(models, X_train, y_train, X_test_non_comorbid, y_test_non_comorbid, X_test_random, y_test_random, output_path):
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_non_comorbid_scaled = scaler.transform(X_test_non_comorbid)
   X_test_random_scaled = scaler.transform(X_test_random)

   results = []
   for name, model in models.items():
       model.fit(X_train_scaled, y_train)

       y_pred_train = model.predict(X_train_scaled)
       y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
       train_metrics = calculate_performance_metrics(y_train, y_pred_train, y_pred_proba_train)

       y_pred_non_comorbid = model.predict(X_test_non_comorbid_scaled)
       y_pred_proba_non_comorbid = model.predict_proba(X_test_non_comorbid_scaled)[:, 1]
       non_comorbid_metrics = calculate_performance_metrics(y_test_non_comorbid, y_pred_non_comorbid, y_pred_proba_non_comorbid)

       y_pred_random = model.predict(X_test_random_scaled)
       y_pred_proba_random = model.predict_proba(X_test_random_scaled)[:, 1]
       random_metrics = calculate_performance_metrics(y_test_random, y_pred_random, y_pred_proba_random)

       # Export outcome variable categories and predicted probabilities to CSV files
       for dataset, y_true, y_pred_proba in zip(['train', 'non_comorbid', 'random'],
                                                [y_train, y_test_non_comorbid, y_test_random],
                                                [y_pred_proba_train, y_pred_proba_non_comorbid, y_pred_proba_random]):
           output_df = pd.DataFrame({
               'Outcome': y_true,
               'Predicted Probability': y_pred_proba
           })
           output_df.to_csv(f"{output_path}/{name}_{dataset}_output.csv", index=False)

       results.extend([
           {'Dataset': 'Train', 'Model': name, 'AUC': train_metrics['AUC'], 'AUPRC': train_metrics['AUPRC']},
           {'Dataset': 'Non-comorbid', 'Model': name, 'AUC': non_comorbid_metrics['AUC'], 'AUPRC': non_comorbid_metrics['AUPRC']},
           {'Dataset': 'Random', 'Model': name, 'AUC': random_metrics['AUC'], 'AUPRC': random_metrics['AUPRC']}
       ])

   # Export model performance metrics to a CSV file
   results_df = pd.DataFrame(results)
   results_df.to_csv(f"{output_path}/model_performance_metrics.csv", index=False)

   return results_df

# Main file 3-18-24
import os
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# from block1 import preprocess_data
# from block2 import calculate_clinical_outcomes
# from block3 import split_data
# from block4 import additional_preprocessing, classify_columns_filtered
# from block5 import add_risk_scores_to_datasets
# from block6 import generate_summary_tables
# from block7 import generate_missing_values_and_vif_correlation
# from block8 import preprocess_datasets
# from block9 import perform_smote
# from block10 import scale_encode_features
# from block11 import initialize_models, perform_cross_validation, evaluate_models
# from block12 import plot_prob_counts

MY_PATH = "/Users/ander428/Documents/Github/Feature-importance-with-fairness-mitigation/data/"

# Path for the input and output CSV file
input_file_path = '/Users/ander428/Documents/Github/Feature-importance-with-fairness-mitigation/data/afib_revised_082025.csv'
output_file_path = MY_PATH + 'temp/preprocessed_data_block1.csv'

# Select and save the columns to be included in the data and save it to a file
df_filtered = preprocess_data(input_file_path, output_file_path)
df_filtered.to_csv(output_file_path, index=False)
print("Block1 selected columns dataset generated and saved")

# Define, calculate, and save the desired clinical outcome and timepoint
outcome = 'MBE' # Change this to the desired outcome (e.g., 'STEMIOrRevascforCAD', 'IstrokeOrTIAOrSTEMIOrRevascforCAD', 'MBE', 'IstrokeOrTIA', 'H_STROKE_READMIT', 'Mortality')
timepoint = '1year' # Change this to the desired timepoint ('1year', '2year', or '5year')
outcome_var = f'{outcome}_{timepoint}'

df_filtered = calculate_clinical_outcomes(df_filtered, outcome, timepoint)
output_file_path_block2 =  MY_PATH + 'temp/preprocessedwithoutcome_data_block2.csv'
df_filtered.to_csv(output_file_path_block2, index=False)
print("Block2 clinical outcome created and saved.")

# Define the stratification factors for stratified data splitting and data percentages; random_state=42
stratify_cols = ['FEMALE', 'Black_Race']
train_pct = 0.7
non_comorbid_pct = 0.5
random_pct = 0.5

# Split the data and export the train and test sets into user-defined pathways
train_set, test_set_non_comorbid, test_set_random = split_data(df_filtered, outcome, timepoint, stratify_cols, train_pct, non_comorbid_pct, random_pct)
train_set_path =  MY_PATH + 'temp/nopreprocessedtrain_set_block3.csv'
test_set_non_comorbid_path =  MY_PATH + 'temp/nopreprocessedtest_set_non_comorbid_block3.csv'
test_set_random_path = MY_PATH + 'temp/nopreprocessedtest_set_random_block3.csv'
train_set.to_csv(train_set_path, index=False)
test_set_non_comorbid.to_csv(test_set_non_comorbid_path, index=False)
test_set_random.to_csv(test_set_random_path, index=False)
print("Block3 data split performed and saved.")

# Perform and export additional preprocessing by feature engineering to create more columns, perform one-hot encoding and scaling the columns on the datasets
preprocessed_train_set, preprocessed_test_set_non_comorbid, preprocessed_test_set_random, categorical_cols_train, continuous_cols_train, categorical_cols_non_comorbid, continuous_cols_non_comorbid, categorical_cols_random, continuous_cols_random = additional_preprocessing(train_set, test_set_non_comorbid, test_set_random)
preprocessed_train_set_path = MY_PATH + 'temp/preprocessed_train_set_block4.csv'
preprocessed_test_set_non_comorbid_path = MY_PATH + 'temp/preprocessed_test_set_non_comorbid_block4.csv'
preprocessed_test_set_random_path = MY_PATH + 'temp/preprocessed_test_set_random_block4.csv'

preprocessed_train_set.to_csv(preprocessed_train_set_path, index=False)
preprocessed_test_set_non_comorbid.to_csv(preprocessed_test_set_non_comorbid_path, index=False)
preprocessed_test_set_random.to_csv(preprocessed_test_set_random_path, index=False)

print("Block4 feature engineering with new column creation and scaling of continuous columns performed and saved.")

# Classify columns in each dataset
categorical_cols_train, continuous_cols_train = classify_columns_filtered(preprocessed_train_set)
categorical_cols_non_comorbid, continuous_cols_non_comorbid = classify_columns_filtered(preprocessed_test_set_non_comorbid)
categorical_cols_random, continuous_cols_random = classify_columns_filtered(preprocessed_test_set_random)

# Compute risk scores, add them to the datasets, and export the updated datasets
preprocessed_train_set, preprocessed_test_set_non_comorbid, preprocessed_test_set_random, continuous_cols_train, continuous_cols_non_comorbid, continuous_cols_random = add_risk_scores_to_datasets(preprocessed_train_set, preprocessed_test_set_non_comorbid, preprocessed_test_set_random, categorical_cols_train, continuous_cols_train, categorical_cols_non_comorbid, continuous_cols_non_comorbid, categorical_cols_random, continuous_cols_random)
preprocessed_train_set_with_risk_scores_path = MY_PATH + 'temp/preprocessed_train_set_block5.csv'
preprocessed_test_set_non_comorbid_with_risk_scores_path = MY_PATH + 'temp/preprocessed_test_set_non_comorbid_block5.csv'
preprocessed_test_set_random_with_risk_scores_path = MY_PATH + 'temp/preprocessed_test_set_random_block5.csv'

preprocessed_train_set.to_csv(preprocessed_train_set_with_risk_scores_path, index=False)
preprocessed_test_set_non_comorbid.to_csv(preprocessed_test_set_non_comorbid_with_risk_scores_path, index=False)
preprocessed_test_set_random.to_csv(preprocessed_test_set_random_with_risk_scores_path, index=False)

print("Block5 HASBLED, ORBIT, ATRIA, CHADSVASc risk scores added and saved.")

# Perform statistical analysis and summarize according to manuscript table standards
summary_table_train_path = MY_PATH + 'temp/summary_statistics_train.csv'
summary_table_test_non_comorbid_path = MY_PATH + 'temp/summary_statistics_test_non_comorbid.csv'
summary_table_test_random_path = MY_PATH + 'temp/summary_statistics_test_random.csv'

generate_summary_tables(train_set, test_set_non_comorbid, test_set_random,
                        categorical_cols_train, continuous_cols_train,
                        categorical_cols_non_comorbid, continuous_cols_non_comorbid,
                        categorical_cols_random, continuous_cols_random,
                        summary_table_train_path, summary_table_test_non_comorbid_path, summary_table_test_random_path,
                        outcome, timepoint)
print("Block6 summary tables according to outcome of interest have been generated and saved.")

# Computation of missing values, VIF and correlation and saving them as files
missing_values_train_path = MY_PATH + 'temp/missing_values_train.csv'
missing_values_test_non_comorbid_path = MY_PATH + 'temp/missing_values_test_non_comorbid.csv'
missing_values_test_random_path = MY_PATH + 'temp/missing_values_test_random.csv'
vif_correlation_train_path = MY_PATH + 'temp/vif_correlation_train.csv'
high_corr_categorical_train_path = MY_PATH + 'temp/high_corr_categorical_train.csv'

generate_missing_values_and_vif_correlation(preprocessed_train_set, preprocessed_test_set_non_comorbid, preprocessed_test_set_random,
                                            categorical_cols_train, continuous_cols_train,
                                            missing_values_train_path, missing_values_test_non_comorbid_path,
                                            missing_values_test_random_path, vif_correlation_train_path,
                                            high_corr_categorical_train_path, corr_threshold=0.8)

print("Block7 missing value, VIF and correlation tables have been generated and saved.")

# Block 8: Further dropping columns not to be included in the model training followed by imputation
preprocessed_train_set_path = MY_PATH + 'temp/preprocessed_train_set_block8.csv'
preprocessed_test_set_non_comorbid_path = MY_PATH + 'temp/preprocessed_test_set_non_comorbid_block8.csv'
preprocessed_test_set_random_path = MY_PATH + 'temp/preprocessed_test_set_random_block8.csv'

train_set_imputed, test_set_non_comorbid_imputed, test_set_random_imputed, categorical_cols_train, continuous_cols_train, categorical_cols_non_comorbid, continuous_cols_non_comorbid, categorical_cols_random, continuous_cols_random = preprocess_datasets(
    preprocessed_train_set, preprocessed_test_set_non_comorbid, preprocessed_test_set_random,
    categorical_cols_train, continuous_cols_train,
    categorical_cols_non_comorbid, continuous_cols_non_comorbid,
    categorical_cols_random, continuous_cols_random,
    high_corr_categorical_train_path,
    preprocessed_train_set_path, preprocessed_test_set_non_comorbid_path, preprocessed_test_set_random_path
)

print("Block8 column dropping and imputation performed.")

# Block 9: Performing SMOTE on the imputed training dataset
smote_train_set_path = MY_PATH + 'temp/smote_train_set_block9.csv'
smote_train_set_missing_path = MY_PATH + 'temp/smote_train_set_missing_block9.csv'

# Perform SMOTE on the imputed training set 
train_set_resampled = perform_smote(train_set_imputed, categorical_cols_train, outcome_var, smote_train_set_path, smote_train_set_missing_path)
positive_percentage = (train_set_resampled[outcome_var].sum() / len(train_set_resampled)) * 100
print(f"Percentage of positive cases in the SMOTE-enhanced dataset: {positive_percentage:.2f}%")
print("Block9 SMOTE-enhanced training set generated.")

# Block 10: Feature scaling and target encoding
scaled_train_set_path = MY_PATH + 'afib_revised_082025_scaled_train_set_block10.csv'
scaled_test_set_non_comorbid_path = MY_PATH + 'temp/scaled_test_set_non_comorbid_block10.csv'
scaled_test_set_random_path = MY_PATH + 'temp/scaled_test_set_random_block10.csv'

train_set_scaled_encoded, test_set_non_comorbid_scaled_encoded, test_set_random_scaled_encoded = scale_encode_features(
    train_set_resampled, test_set_non_comorbid_imputed, test_set_random_imputed,
    categorical_cols_train, continuous_cols_train, outcome_var,
    scaled_train_set_path, scaled_test_set_non_comorbid_path, scaled_test_set_random_path
)
print("Block10 feature scaling and target encoding completed.")
print(train_set_scaled_encoded)