# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
from helper_functions.data_cleaning import data_cleaning
from helper_functions.visualisations import visualisations
from helper_functions.dataset import dataset
from helper_functions.iqr import iqr
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, MaxAbsScaler, FunctionTransformer
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
Please accompanying 'INFOSYS 722 Iteration 03 : STEPS 1 - 8 (OSAS)_(mmut001, 9564814)' document
for further documnation of this programme

NOTE: If running this in VSCode, please ensure the Figure files generated to visulaize the data are closed as the appear 
to permit the programme to continue to run

"""
dc = data_cleaning()
v = visualisations()
iqr= iqr()
ds = dataset()

FACTOR = 3.0


# 01-BU

'''
Please see Section 2 in the accompanying report for further details of the business understanding
'''


# 02-DU
'''
Complete data understanding steps to gain a deeper understanding of the data
'''

# ---------- * Section 2.3: Data Exploration in accompanying report * ----------
# Explore Data
# Load Datasets and ensure data types are as expected
g_f_db_2025_csv = 'GlobalFindexDatabase2025.csv'
df_g_f_db_2025 = pd.read_csv(g_f_db_2025_csv, dtype={
    'countrynewwb': str, 'codewb': str, 'year': int, 'pop_adult': float, 'regionwb24_hi': str, 'incomegroupwb24': str, 'group2': str, 'account.t.d': float, 
    'fiaccount.t.d': float, 'mobileaccount.t.d': float, 'borrow.any.td': float, 'fin17a.17a1.d': float, 'fin22b': float, 'fin24aP': float, 'fin24aN': float, 
    'fin30': float, 'fin32.n33': float, 'fin32': float, 'fin32.acc': float, 'fin37.38': float, 'fin2.t.d': float, 'fin10': float, 'fing2p': float, 
    'g20.made': float, 'g20.received': float, 'g20.any': float, 'save.any.t.d': float, 'con1': float
    })

hdr_composite_csv = 'HDR25_Composite_indices_complete_time_series.csv'
df_hdr_composite_2025 = pd.read_csv(hdr_composite_csv, dtype={
    'iso3': str, 'country': str, 'hdicode': str, 'region': str, 'gdi_2011': float, 'gdi_2014': float, 'gdi_2017': float, 'gdi_2021': float, 
    'hdi_f_2011': float, 'hdi_f_2014': float, 'hdi_f_2017': float, 'hdi_f_2021': float, 'mys_f_2011': float, 'mys_f_2014': float, 
    'mys_f_2017': float, 'mys_f_2021': float, 'gni_pc_f_2011': float, 'gni_pc_f_2014': float, 'gni_pc_f_2017': float, 'gni_pc_f_2021': float,
    'hdi_m_2011': float, 'hdi_m_2014': float, 'hdi_m_2017': float,  'hdi_m_2021': float, 'eys_m_2011': float, 'eys_m_2014': float, 'eys_m_2017': float, 
    'eys_m_2021': float, 'mys_m_2011': float, 'mys_m_2014': float, 'mys_m_2017': float, 'mys_m_2021': float, 'gni_pc_m_2011': float, 'gni_pc_m_2014': float, 
    'gni_pc_m_2017': float, 'gni_pc_m_2021': float, 'ineq_edu_2011': float, 'ineq_edu_2014': float, 'ineq_edu_2017': float, 'ineq_edu_2021': float, 
    'ineq_inc_2011': float, 'ineq_inc_2014': float, 'ineq_inc_2017': float, 'ineq_inc_2021': float
    })

world_regions_2023_csv = 'world-regions-according-to-the-world-bank.csv'
df_world_regions_2023 = pd.read_csv(world_regions_2023_csv, dtype={ 'Entity': str, 'Code': str, 'Year': int, 'World regions according to WB': str })

print('---------- *\nSection 2.3: df_g_f_db_2025 dataset information:')
print(df_g_f_db_2025.info(), '\n')

print('---------- *\nSection 2.3: df_hdr_composite_2025 dataset information:')
print(df_hdr_composite_2025.info(), '\n')

print('---------- *\nSection 2.3: df_world_regions_2023 dataset information:')
print(df_world_regions_2023.info(), '\n')

# ---------- * Section 2.4: Verify the Data Quality in accompanying report * ----------
# Rename the country features across all datasets to allow merging and lookup of datasets
df_g_f_db_2025.rename(columns={'countrynewwb': 'country', 'codewb': 'code'}, inplace=True)
df_world_regions_2023.rename(columns={'Entity': 'country', 'Code': 'code', 'Year': 'year', 'World regions according to WB': 'region'}, inplace=True)
df_hdr_composite_2025.rename(columns={'iso3': 'code'}, inplace=True)

# Rename the GlobalFindexDatabase2025 features for readability
df_g_f_db_2025.rename(columns={
    'pop_adult': 'population', 'regionwb24_hi': 'region', 'incomegroupwb24': 'income_group', 'group2': 'gender', 'account.t.d': 'account',
    'fiaccount.t.d': 'bank_account', 'mobileaccount.t.d': 'mobile_account', 'borrow.any.t.d': 'borrow_01', 'fin17a.17a1.d': 'saved_money_at_bank', 
    'fin22b': 'borrow_02', 'fin24aP': 'emergency_01', 'fin24aN': 'emergency_02', 'fin30': 'utility', 'fin32.n33': 'wages_01', 'fin32': 'wages_02',
    'fin32.acc': 'wages_03', 'fin37.38': 'pension', 'fin2.t.d': 'debit_card', 'fin10': 'credit_card', 'fing2p': 'government_pay', 
    'g20.made': 'made_digital_payment', 'g20.received': 'received_digital_payment', 'g20.any': 'any_digital_payment', 'save.any.t.d': 'save_any_money', 
    'con1': 'own_mobile_phone'
    }, inplace=True)

# Remove the GlobalFindexDatabase2025 dataframe features that are not under analysis
df_g_f_db_2025_columns = [
    'country', 'code', 'year', 'population', 'region', 'income_group', 'gender', 'account','bank_account', 'mobile_account', 'borrow_01','saved_money_at_bank', 
    'borrow_02', 'emergency_01', 'emergency_02', 'utility', 'wages_01', 'wages_02', 'wages_03', 'pension', 'debit_card', 'credit_card', 'government_pay', 
    'made_digital_payment', 'received_digital_payment', 'any_digital_payment', 'save_any_money', 'own_mobile_phone'
    ]
df_g_f_db_2025.drop(columns=[col for col in df_g_f_db_2025.columns if col not in df_g_f_db_2025_columns], axis = 1, inplace = True)

print('---------- *\nSection 3.2.1: Updated df_g_f_db_2025 dataset information')
print(df_g_f_db_2025.info(), '\n')

# Remove records from the GlobalFindexDatabase2025 dataframe where country names are blank as associated codes do not match any known country code
df_g_f_db_2025 = df_g_f_db_2025.dropna(subset=['country'])

# Remove records from the GlobalFindexDatabase2025 dataframe where country codes do not align with df_hdr_composite_2025
remove_codes = ['XKX', 'TWN', 'PRI', 'SAS']
df_g_f_db_2025 = df_g_f_db_2025[~df_g_f_db_2025['code'].isin(remove_codes)]
print('---------- *\nSection 2.3: df_g_f_db_2025 dataset information AFTER country name and code clean up:')
print(df_g_f_db_2025.info(), '\n')


# Remove records from the GlobalFindexDatabase2025 dataframe that are not 'men' or 'women' gender records
selected_cols = ['gender']
print('---------- *\nSection 2.3: df_g_f_db_2025 dataset information BEFORE gender filter:')
print(df_g_f_db_2025[selected_cols].info(), '\n')
genders = ['men', 'women']
df_g_f_db_2025 = df_g_f_db_2025[df_g_f_db_2025['gender'].isin(genders)]
print('---------- *\nSection 2.3: df_g_f_db_2025 dataset information AFTER gender filter:')
print(df_g_f_db_2025[selected_cols].info(), '\n')

# Add Visualisations
# Show GlobalFindexDatabase2025 dataframe gender distribution
#
# NOTE: If running this in VSCode, please ensure the Figure files generated are closed to permit programme to continue
v.show_horizontal_histogram(df_g_f_db_2025, 'gender', 'GlobalFindexDatabase2025')

# Show GlobalFindexDatabase2025 dataframe dot chart of gender against income_group & own_mobile_phone
#
# NOTE: If running this in VSCode, please ensure the Figure files generated are closed to permit programme to continue
v.show_dot_chart(df_g_f_db_2025, 'income_group', 'own_mobile_phone', 'gender')

# Replace GlobalFindexDatabase2025 dataset region feature to use region specified in df_world_regions_2023. Resolves  
# records that have 'High income' in the region field
df_g_f_db_2025 = dc.update_regions(df_world_regions_2023, df_g_f_db_2025)

# Update GlobalFindexDatabase2025 dataframe to use df_world_regions_2023 
# country names
dc.update_country_names(df_world_regions_2023, df_g_f_db_2025)
print('---------- *\nSection 3.2.2: Updated df_g_f_db_2025 dataset information')
print(df_g_f_db_2025['country'].describe())
print(df_g_f_db_2025['country'].unique()[:20], '\n')

# Check if any "High income" values remain
count_high_income = (df_g_f_db_2025['region'].str.strip() == 'High income').sum()

print('---------- *\nSection 2.4.1: df_g_f_db_2025 dataset information AFTER updating regions:')
if count_high_income == 0:
    print("✅ No 'High income' values remain in the 'region' column.\n")
else:
    print(f"⚠️ Found {count_high_income} rows with 'High income' in the 'region' column.\n")

print('---------- *\nSection 3.3.1: Updated df_g_f_db_2025 dataset information:')
print(df_g_f_db_2025.info(), '\n')

# Rename the df_hdr_composite_2025 features for readability
df_hdr_composite_2025.rename(columns={
    'hdicode': 'human_development_groups_level', 'gdi_2011': 'gender_development_index_2011', 'gdi_2014': 'gender_development_index_2014', 
    'gdi_2017': 'gender_development_index_2017', 'gdi_2021': 'gender_development_index_2021', 'gdi_2022': 'gender_development_index_2022', 
    'hdi_f_2011': 'human_development_index_female_2011', 'hdi_f_2014': 'human_development_index_female_2014', 
    'hdi_f_2017': 'human_development_index_female_2017', 'hdi_f_2021': 'human_development_index_female_2021', 
    'hdi_f_2022': 'human_development_index_female_2022', 'mys_f_2011': 'mean_years_schooling_female_2011', 
    'mys_f_2014': 'mean_years_schooling_female_2014', 'mys_f_2017': 'mean_years_schooling_female_2017', 
    'mys_f_2021': 'mean_years_schooling_female_2021', 'mys_f_2022': 'mean_years_schooling_female_2022', 
    'gni_pc_f_2011': 'gross_national_income_per_capita_female_2011', 'gni_pc_f_2014': 'gross_national_income_per_capita_female_2014', 
    'gni_pc_f_2017': 'gross_national_income_per_capita_female_2017', 'gni_pc_f_2021': 'gross_national_income_per_capita_female_2021',
    'gni_pc_f_2022': 'gross_national_income_per_capita_female_2022', 'hdi_m_2011': 'human_development_index_male_2011', 
    'hdi_m_2014': 'human_development_index_male_2014', 'hdi_m_2017': 'human_development_index_male_2017',  
    'hdi_m_2021': 'human_development_index_male_2021', 'hdi_m_2021': 'human_development_index_male_2021', 
    'hdi_m_2022': 'human_development_index_male_2022', 'mys_m_2011': 'mean_years_schooling_male_2011', 
    'mys_m_2014': 'mean_years_schooling_male_2014', 'mys_m_2017': 'mean_years_schooling_male_2017', 
    'mys_m_2021': 'mean_years_schooling_male_2021', 'mys_m_2022': 'mean_years_schooling_male_2022',
    'gni_pc_m_2011': 'gross_national_income_per_capita_male_2011', 'gni_pc_m_2014': 'gross_national_income_per_capita_male_2014', 
    'gni_pc_m_2017': 'gross_national_income_per_capita_male_2017', 'gni_pc_m_2021': 'gross_national_income_per_capita_male_2021', 
    'gni_pc_m_2022': 'gross_national_income_per_capita_male_2022', 'ineq_edu_2011': 'inequality_in_education_2011', 
    'ineq_edu_2014': 'inequality_in_education_2014', 'ineq_edu_2017': 'inequality_in_education_2017', 
    'ineq_edu_2021': 'inequality_in_education_2021', 'ineq_edu_2022': 'inequality_in_education_2022', 
    'ineq_inc_2011': 'inequality_in_income_2011', 'ineq_inc_2014': 'inequality_in_income_2014', 
    'ineq_inc_2017': 'inequality_in_income_2017', 'ineq_inc_2021': 'inequality_in_income_2021', 'ineq_inc_2022': 'inequality_in_income_2022'
    }, inplace=True)

# Remove the HDR25_Composite_indices_complete_time_series dataframe features that are not under analysis
df_hdr_composite_2025_columns = [
    'code', 'country', 'human_development_groups_level', 'region', 'gender_development_index_2011', 'gender_development_index_2014', 'gender_development_index_2017', 
    'gender_development_index_2021', 'gender_development_index_2022', 'human_development_index_female_2011', 'human_development_index_female_2014', 
    'human_development_index_female_2017', 'human_development_index_female_2021', 'human_development_index_female_2022','mean_years_schooling_female_2011', 
    'mean_years_schooling_female_2014', 'mean_years_schooling_female_2017', 'mean_years_schooling_female_2021', 'mean_years_schooling_female_2022',
    'gross_national_income_per_capita_female_2011', 'gross_national_income_per_capita_female_2014', 'gross_national_income_per_capita_female_2017', 
    'gross_national_income_per_capita_female_2021', 'gross_national_income_per_capita_female_2022', 'human_development_index_male_2011', 
    'human_development_index_male_2014', 'human_development_index_male_2017',  'human_development_index_male_2021', 'human_development_index_male_2022', 
    'mean_years_schooling_male_2011', 'mean_years_schooling_male_2014', 'mean_years_schooling_male_2017', 'mean_years_schooling_male_2021', 
    'mean_years_schooling_male_2022', 'gross_national_income_per_capita_male_2011', 'gross_national_income_per_capita_male_2014', 
    'gross_national_income_per_capita_male_2017', 'gross_national_income_per_capita_male_2021', 'gross_national_income_per_capita_male_2022', 
    'inequality_in_education_2011', 'inequality_in_education_2014', 'inequality_in_education_2017', 'inequality_in_education_2021', 'inequality_in_education_2022',
    'inequality_in_income_2011', 'inequality_in_income_2014', 'inequality_in_income_2017', 'inequality_in_income_2021', 'inequality_in_income_2022']
df_hdr_composite_2025.drop(columns=[col for col in df_hdr_composite_2025.columns if col not in df_hdr_composite_2025_columns], axis = 1, inplace = True)

print('---------- *\nSection 3.2.1: Updated df_hdr_composite_2025 dataset information')
print(df_hdr_composite_2025.info(), '\n')

# Update HDR25_Composite_indices_complete_time_series dataframe records to use same regions 
# as GlobalFindexDatabase2025 dataframe
df_hdr_composite_2025 = dc.update_regions(df_world_regions_2023, df_hdr_composite_2025)
print('---------- *\nSection 2.4: Updated df_hdr_composite_2025 dataset information')
print(df_hdr_composite_2025.info(), '\n')

# Update HDR25_Composite_indices_complete_time_series dataframe to use df_world_regions_2023 
# country names
dc.update_country_names(df_world_regions_2023, df_hdr_composite_2025)
print('---------- *\nSection 3.2.2: Updated df_hdr_composite_2025 dataset information')
print(df_hdr_composite_2025['country'].describe())
print(df_hdr_composite_2025['country'].unique()[:25], '\n')

# Replace HDR25_Composite_indices_complete_time_series dataframe records that have 'Other Countries or Territories' with 'High' 
# in human_development_groups_level
df_hdr_composite_2025.loc[
    df_hdr_composite_2025['human_development_groups_level'] == 'Other Countries or Territories',
    'human_development_groups_level'
] = 'High'

# Remove records from HDR25_Composite_indices_complete_time_series dataframe where country code is not recognised
remove_codes = [
    'ZZA.VHHD', 'ZZB.HHD', 'ZZC.MHD', 'ZZD.LHD', 'ZZE.AS',
    'ZZF.EAP', 'ZZG.ECA', 'ZZH.LAC', 'ZZI.SA', 'ZZJ.SSA', 'ZZK.WORLD'
]
df_hdr_composite_2025 = df_hdr_composite_2025[~df_hdr_composite_2025['code'].isin(remove_codes)]

print('---------- *\nSection 3.3.1/Section 3.5: Updated df_hdr_composite_2025 dataset information')
print(df_hdr_composite_2025.info(), '\n')

# 03-DP
'''
Complete data preparation step
'''
# ---------- * Section 3.3.1 Null and Missing Values in accompanying report * ----------
########
# Handle null and missing values in the GlobalFindexDatabase2025 & df_hdr_composite_2025 dataframes
########

# Impute null values in numeric columns in GlobalFindexDatabase2025 dataframe with the mean value
numeric_imputer = SimpleImputer(strategy='mean')
df_g_f_db_2025_numeric_cols = df_g_f_db_2025.select_dtypes(include='number').columns
df_g_f_db_2025[df_g_f_db_2025_numeric_cols] = numeric_imputer.fit_transform(
        df_g_f_db_2025[df_g_f_db_2025_numeric_cols]
    )

# Impute null values in categorical columns in GlobalFindexDatabase2025 dataframe with the most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_cols = df_g_f_db_2025.select_dtypes(include='object').columns
df_g_f_db_2025[categorical_cols] = categorical_imputer.fit_transform(
        df_g_f_db_2025[categorical_cols]
    )
print('---------- *\nSection 3.3.1: Updated df_g_f_db_2025 dataset information:')
print(df_g_f_db_2025.info(), '\n')

# Impute null values in numeric columns in df_hdr_composite_2025 dataframe with the mean value
df_hdr_composite_2025_numeric_cols = df_hdr_composite_2025.select_dtypes(include='number').columns
df_hdr_composite_2025[df_hdr_composite_2025_numeric_cols] = numeric_imputer.fit_transform(
        df_hdr_composite_2025[df_hdr_composite_2025_numeric_cols]
    )
print('---------- *\nSection 3.3.1: Updated df_hdr_composite_2025 dataset information:')
print(df_hdr_composite_2025.info(), '\n')

# ---------- * Section 3.2.2: Extreme and Outlier Values in accompanying report * ----------
# Add Visualisations
###
# Boxplots of GlobalFindexDatabase2025 & df_hdr_composite_2025 features with outlier values
#
# NOTE: If running this in VSCode, please ensure the Figure files generated are closed to permit programme to continue
###
print('---------- *\nSection 3.3.2: Columns with outliers', '\n')
# Grid of only the columns that actually have outliers
df_g_f_db_2025_outliers = v.show_boxplots_with_outliers(df_g_f_db_2025, 'GlobalFindexDatabase2025', FACTOR)
print('Plotted columns:', df_g_f_db_2025_outliers, '\n')
v.show_all_histograms_with_outliers(df_g_f_db_2025, 'GlobalFindexDatabase2025', FACTOR)

exclude_cols = ['gender_binary', 'income_group_binary', 'year', 'own_mobile_phone']  # example

v.show_boxplots_with_outliers(df_hdr_composite_2025, 'df_hdr_composite_2025', FACTOR)
v.show_all_histograms_with_outliers(df_hdr_composite_2025, 'df_hdr_composite_2025', FACTOR)

########
# Handle extreme and outlier values in the GlobalFindexDatabase2025 & df_hdr_composite_2025 dataframes
########
# Cap extremes to IQR bounds (keep all rows)
df_g_f_db_2025_capped, df_g_f_db_2025_iqr_bnds = iqr.handle_outliers_iqr(df_g_f_db_2025, factor=1.5, strategy='cap')
df_hdr_composite_2025_capped, df_hdr_composite_2025_iqr_bnds = iqr.handle_outliers_iqr(df_hdr_composite_2025, factor=1.5, strategy='cap')

print('---------- *\nSection 3.2.2: df_g_f_db_2025_capped dataset information:')
print(df_g_f_db_2025_capped.info(), '\n')
print('---------- *\nSection 3.2.2: df_hdr_composite_2025_capped dataset information:')
print(df_hdr_composite_2025_capped.info(), '\n')

# ---------- * Section 3.3: Data Construction in accompanying report * ----------
# Merge income groups from the GlobalFindexDatabase2025 dataframe into two categories
df_g_f_db_2025['income_group_binary_label'] = df_g_f_db_2025['income_group'].replace({
    'High income': 'High income',
    'Upper middle income': 'High income',
    'Low income': 'Low income',
    'Lower middle income': 'Low income'
})

# Convert income groups from the GlobalFindexDatabase2025 dataframe to binary (High income=1, Low income=0)
df_g_f_db_2025['income_group_binary'] = df_g_f_db_2025['income_group_binary_label'].map({
    'Low income': 0,
    'High income': 1
})

print(df_g_f_db_2025[['income_group', 'income_group_binary_label', 'income_group_binary']].head(10), '\n')

# Convert gender from the GlobalFindexDatabase2025 dataframe into binary (men=0, women=1)
df_g_f_db_2025['gender_binary'] = df_g_f_db_2025['gender'].str.strip().str.lower().map({
    'men': 0,
    'women': 1
})

# Merge human_development_groups_level from the df_hdr_composite_2025 dataframe into two categories
df_hdr_composite_2025['human_development_groups_level_binary_label'] = df_hdr_composite_2025['human_development_groups_level'].replace({
    'Very High': 'High',
    'High': 'High',
    'Medium': 'Low',
    'Low': 'Low'
})

# Convert human_development_groups_level from the df_hdr_composite_2025 dataframe to binary (High=1, Low=0)
df_hdr_composite_2025['human_development_groups_level_binary'] = df_hdr_composite_2025['human_development_groups_level_binary_label'].map({
    'Low': 0,
    'High': 1
})

stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
out_path = f'{stamp}_df_g_f_db_2025.csv'
df_g_f_db_2025.to_csv(out_path, index=False, encoding='utf-8', na_rep='')
print(f'Saved merged dataset to: {out_path}', '\n')
print('---------- *\nSection 3.3: Updated df_g_f_db_2025 dataset information:')
print(df_g_f_db_2025.info(), '\n')
out_path = f'{stamp}_df_hdr_composite_2025.csv'
df_hdr_composite_2025.to_csv(out_path, index=False, encoding='utf-8', na_rep='')
print('---------- *\nSection 3.3: Updated df_hdr_composite_2025 dataset information:')
print(df_hdr_composite_2025.info(), '\n')

# ---------- * Section 3.4: Data Integration in accompanying report * ----------
df_g_f_db_2025 =ds. merge_datasets(df_g_f_db_2025, 'df_g_f_db_2025', df_hdr_composite_2025, 'df_hdr_composite_2025')
stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
out_path = f'{stamp}_df_g_f_db_2025.csv'
df_g_f_db_2025.to_csv(out_path, index=False, encoding='utf-8', na_rep='')

df_g_f_db_2025_numeric_cols = df_g_f_db_2025.select_dtypes(include='number').columns
df_g_f_db_2025[df_g_f_db_2025_numeric_cols] = numeric_imputer.fit_transform(df_g_f_db_2025[df_g_f_db_2025_numeric_cols])
categorical_cols = df_g_f_db_2025.select_dtypes(include='object').columns
df_g_f_db_2025[categorical_cols] = categorical_imputer.fit_transform(df_g_f_db_2025[categorical_cols])
print('---------- *\nSection 3.4: Updated df_g_f_db_2025 dataset information:')
print(df_g_f_db_2025.info(), '\n')


# 04-DT
'''
Complete data transformation step
'''
# ---------- * Section 4.1 : Data Reduction in accompanying report * ----------
# Scale/normalize all features consistently on own_mobile_phone
# --- 0) Split target / features ---
y = df_g_f_db_2025['own_mobile_phone']
X = df_g_f_db_2025.drop(columns=['own_mobile_phone'])

# --- 1) Basic hygiene: replace infs and drop rows where y is not finite ---
X = X.replace([np.inf, -np.inf], np.nan)
finite_mask = np.isfinite(y)
X, y = X.loc[finite_mask].copy(), y.loc[finite_mask].copy()

# --- 2) Column typing ---
num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(exclude='number').columns.tolist()

# ---------------------------
# Preprocess pipelines
# ---------------------------
num_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    # Quantile -> approximate Normal(0,1); neutralizes heavy tails/outliers
    ('qt', QuantileTransformer(n_quantiles=1000, output_distribution='normal', subsample=1_000_000, random_state=42)),
])

cat_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ],
    remainder='drop'
)

# Safety: ensure finite values, then bound magnitudes
nan_inf_guard = FunctionTransformer(lambda A: np.nan_to_num(A, nan=0.0, posinf=1e3, neginf=-1e3))
bound_scale   = MaxAbsScaler()  # ensures max |value| per feature == 1

model = Ridge(alpha=50.0, random_state=42)  # or: HuberRegressor(epsilon=1.35, alpha=0.0001)

pipe = Pipeline(steps=[
    ('prep', preprocess),
    ('finite', nan_inf_guard),
    ('bound', bound_scale),
    ('model', model)
])

# ---------------------------
# Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

pipe.fit(X_train, y_train)

# Quick diagnostics on the transformed train matrix
Xt_train = pipe.named_steps['bound'].transform(
    pipe.named_steps['finite'].transform(
        pipe.named_steps['prep'].transform(X_train)
    )
)
print(
    f"After preprocess: finite={np.isfinite(Xt_train).all()} | "
    f"max_abs={np.abs(Xt_train).max():.3f}"
, '\n')

# Predict + metrics (clip to [0,1] if your target is a proportion)
y_pred = np.clip(pipe.predict(X_test), 0, 1)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f} | R²: {r2:.4f}", '\n')

print('---------- *\nSection 4.1: Updated df_g_f_db_2025 dataset information:')
print(df_g_f_db_2025.info(), '\n')

# --- Determine feature correlation on gender ---
df_g_f_db_2025 = df_g_f_db_2025.dropna(subset=['gender_binary'])

# Compute correlations (now includes gender_bin)
corr = df_g_f_db_2025.corr(numeric_only=True)

print('---------- *\nCorrelation of features with gender (women=1, men=0):')
print(corr['gender_binary'].sort_values(ascending=False), '\n')

# --- Determine feature correlation on own_mobile_phone ---
df_g_f_db_2025 = df_g_f_db_2025.dropna(subset=['own_mobile_phone'])

# Compute correlations (now includes gender_bin)
corr = df_g_f_db_2025.corr(numeric_only=True)

print('---------- *\nCorrelation of features with own_mobile_phone:')
print(corr['own_mobile_phone'].sort_values(ascending=False), '\n')
# Add any transformation steps

# 05-DMM
'''
Identify the Data Mining method
Describe how it aligns with the objectives
'''

# 06-DMA
'''
Identify the Data Mining Algorithms
Describe how it aligns with the objectives
'''
# Load relevant algorithms

# 07-DM
'''
Complete data mining step
'''

# Execute DM task

# 08-INT
'''
Complete resuts and patterns interpretation step
'''

# Summarise Results

# Add relevant tables or graphs

# 09-ACT
'''
Describe the Action Plan to Implement, Observe and Improve
'''
