# -*- coding: utf-8 -*-
from helper_functions.iqr import iqr as IQR
from helper_functions.visualisations import visualisations
from helper_functions.dataset import dataset
from datetime import datetime
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

'''
Please see accompanying 'INFOSYS 722 Iteration 03 : STEPS 1 - 8 (OSAS)_(mmut001, 9564814)' document
for further documentation of this code

NOTE: If running this in VSCode, please ensure the Figure files generated to visulaize the data are 
closed as they appear to permit the code to continue to run

************************************************************************

THE CODE BELOW ONLY DEALS WITH STEP 03 OF THE DATA MINING PROCESS (OSAS) 

************************************************************************

'''

iqr = IQR()
v = visualisations()
ds = dataset()
dataset_names = ['GlobalFindexDatabase2025', 'HDR25_Composite_indices_complete_time_series']
# 01-BU

'''
Please see Section 1 in the accompanying report for further details of the business understanding
'''


# 02-DU
'''
Please see Section 2 in the accompanying report & code in 'iteration_03_step_02_OSAS' for further details
'''

# 03-DP
'''
Complete data preparation step
'''
# ---------- * Section 3.3.1 Null and Missing Values in accompanying report * ----------
class step_03:
    def data_preparation(self, dataframe_fin, dataframe_hdr, FACTOR):
        df_g_f_db_2025          = dataframe_fin
        df_hdr_composite_2025   = dataframe_hdr

        print(f'---------- *\nSection 3.3.1: Updated {dataset_names[0]} dataset information:')
        print(df_g_f_db_2025.info(), '\n')

        print(f'---------- *\nSection 3.3.1: Updated {dataset_names[1]} dataset information:')
        print(df_hdr_composite_2025.info(), '\n')

        # Due to large number of NaN values in own_mobile_phone handle specifically to ensure at least two 
        # classes exisit for SMOTE oversampling in Step 04
        df_g_f_db_2025 = iqr.impute_missing_values(df_g_f_db_2025, 'own_mobile_phone', 0.87)
        print(df_g_f_db_2025['own_mobile_phone'].head(40), '\n')

        ########
        # Handle null and missing values in the GlobalFindexDatabase2025 & HDR25_Composite_indices_complete_time_series dataframes
        ########

        # Impute null values in numeric columns in GlobalFindexDatabase2025 dataframe with the mean value
        numeric_imputer = SimpleImputer(strategy='mean')
        merged_step_03_numeric_cols = df_g_f_db_2025.select_dtypes(include='number').columns
        df_g_f_db_2025[merged_step_03_numeric_cols] = numeric_imputer.fit_transform(
                df_g_f_db_2025[merged_step_03_numeric_cols]
            )

        # Impute null values in categorical columns in GlobalFindexDatabase2025 dataframe with the most frequent value
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        categorical_cols = df_g_f_db_2025.select_dtypes(include='object').columns
        df_g_f_db_2025[categorical_cols] = categorical_imputer.fit_transform(
                df_g_f_db_2025[categorical_cols]
            )
        print(f'---------- *\nSection 3.3.1: Updated {dataset_names[0]} dataset information:')
        print(df_g_f_db_2025.info(), '\n')

        # Impute null values in numeric columns in HDR25_Composite_indices_complete_time_series dataframe with the mean value
        df_hdr_composite_2025_numeric_cols = df_hdr_composite_2025.select_dtypes(include='number').columns
        df_hdr_composite_2025[df_hdr_composite_2025_numeric_cols] = numeric_imputer.fit_transform(
                df_hdr_composite_2025[df_hdr_composite_2025_numeric_cols]
            )
        print(f'---------- *\nSection 3.3.1: Updated {dataset_names[1]} dataset information:')
        print(df_hdr_composite_2025.info(), '\n')

        # ---------- * Section 3.2.2: Extreme and Outlier Values in accompanying report * ----------
        ###
        # Identify features with outliers and display boxplots and histograms of GlobalFindexDatabase2025 & 
        # HDR25_Composite_indices_complete_time_series features with outlier values
        #
        # NOTE: If running this in VSCode, please ensure the Figure files generated are closed to permit code to continue
        ###
        print(f'---------- *\nSection 3.3.2: {dataset_names[0]} features with outliers')
        iqr.find_outliers_iqr(df_g_f_db_2025, dataset_names[0], FACTOR)

        # Grid of only the columns that actually have outliers
        df_g_f_db_2025_outliers = v.show_boxplots_with_outliers(df_g_f_db_2025, dataset_names[0], FACTOR)
        print(f'{dataset_names[0]} features with outliers:')
        v.print_outlier_columns(df_g_f_db_2025_outliers)
        v.show_all_histograms_with_outliers(df_g_f_db_2025, dataset_names[0], FACTOR)

        print(f'---------- *\nSection 3.3.2: {dataset_names[1]} features with outliers')
        iqr.find_outliers_iqr(df_hdr_composite_2025, dataset_names[1], FACTOR)      
        
        # Grid of only the columns that actually have outliers
        df_hdr_composite_2025_outliers = v.show_boxplots_with_outliers(df_hdr_composite_2025, dataset_names[1], FACTOR)
        print(f'{dataset_names[1]} features with outliers:')
        v.print_outlier_columns(df_hdr_composite_2025_outliers)
        v.show_all_histograms_with_outliers(df_hdr_composite_2025, dataset_names[1], FACTOR)

        ########
        # Handle extreme and outlier values in the GlobalFindexDatabase2025 & HDR25_Composite_indices_complete_time_series dataframes
        # by capping extreme values to keep all rows
        ########
        # Cap extremes to IQR bounds (keep all rows)
        df_g_f_db_2025_capped = iqr.handle_outliers_iqr(df_g_f_db_2025, factor=FACTOR, strategy='cap')
        df_hdr_composite_2025_capped = iqr.handle_outliers_iqr(df_hdr_composite_2025, factor=FACTOR, strategy='cap')

        print(f'---------- *\nSection 3.2.2: Capped {dataset_names[0]} dataset information:')
        print(df_g_f_db_2025_capped.info(), '\n')
        print(f'---------- *\nSection 3.2.2: Capped {dataset_names[1]} dataset information:')
        print(df_hdr_composite_2025_capped.info(), '\n')

        # ---------- * Section 3.4: Data Construction in accompanying report * ----------
        # Merge income groups from the GlobalFindexDatabase2025 dataframe into two categories
        df_g_f_db_2025_capped['income_group_binary_label'] = df_g_f_db_2025_capped['income_group'].replace({
            'High income': 'High income',
            'Upper middle income': 'High income',
            'Low income': 'Low income',
            'Lower middle income': 'Low income'
        })

        # Convert income groups from the GlobalFindexDatabase2025 dataframe to binary (High income=1, Low income=0)
        df_g_f_db_2025_capped['income_group_binary'] = df_g_f_db_2025_capped['income_group_binary_label'].map({
            'Low income': 0,
            'High income': 1
        })

        v.show_horizontal_histogram(df_g_f_db_2025_capped, 'income_group_binary', dataset_names[0])
        print(f'---------- *\nSection 3.4: {dataset_names[0]} dataset income_group_binary data construction:')
        print(df_g_f_db_2025_capped[['income_group', 'income_group_binary_label', 'income_group_binary']].head(10), '\n')
        df_g_f_db_2025_capped = df_g_f_db_2025_capped.drop(columns=['income_group_binary_label'])

        # Convert gender from the GlobalFindexDatabase2025 dataframe into binary (men=0, women=1)
        df_g_f_db_2025_capped['gender_binary'] = df_g_f_db_2025_capped['gender'].str.strip().str.lower().map({
            'men': 0,
            'women': 1
        })

        v.show_horizontal_histogram(df_g_f_db_2025_capped, 'gender_binary', dataset_names[0])
        print(f'---------- *\nSection 3.4: {dataset_names[0]} dataset gender_binary data construction:')
        print(df_g_f_db_2025_capped[['gender', 'gender_binary']].head(10), '\n')

        # Merge human_development_groups_level from the HDR25_Composite_indices_complete_time_series dataframe into two categories
        df_hdr_composite_2025_capped[
            'human_development_groups_level_binary_label'
            ] = df_hdr_composite_2025_capped['human_development_groups_level'].replace({
            'Very High': 'High',
            'High': 'High',
            'Medium': 'Low',
            'Low': 'Low'
        })

        # Convert human_development_groups_level from the HDR25_Composite_indices_complete_time_series dataframe to binary (High=1, Low=0)
        df_hdr_composite_2025_capped[
            'human_development_groups_level_binary'
            ] = df_hdr_composite_2025_capped['human_development_groups_level_binary_label'].map({
            'Low': 0,
            'High': 1
        })

        v.show_horizontal_histogram(df_hdr_composite_2025_capped, 'human_development_groups_level_binary', dataset_names[1])
        print(f'---------- *\nSection 3.4: {dataset_names[0]} dataset human_development_groups_level_binary data construction:')
        print(df_hdr_composite_2025_capped[[
            'human_development_groups_level', 
            'human_development_groups_level_binary_label', 
            'human_development_groups_level_binary']].head(10), 
            '\n')
        
        df_hdr_composite_2025_capped = df_hdr_composite_2025_capped.drop(columns=['human_development_groups_level_binary_label'])

        # To permit sanity check of data export to csv
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')

        out_path = f'df_g_f_db_2025_capped_{stamp}.csv'
        df_g_f_db_2025_capped.to_csv(out_path, index=False, encoding='utf-8', na_rep='')
        print(f'---------- *\nSection 3.3: Updated {dataset_names[0]} dataset information:')
        print(f'Saved updated dataset to: {out_path}', '\n')
        print(df_g_f_db_2025_capped.info(), '\n')

        out_path = f'df_hdr_composite_2025_capped_{stamp}.csv'
        df_hdr_composite_2025_capped.to_csv(out_path, index=False, encoding='utf-8', na_rep='')
        print(f'---------- *\nSection 3.3: Updated {dataset_names[1]} dataset information:')
        print(f'Saved updated dataset to: {out_path}', '\n')
        print(df_hdr_composite_2025_capped.info(), '\n')

        # ---------- * Section 3.4: Data Integration in accompanying report * ----------
        merged_step_03 =ds.merge_datasets(
            df_g_f_db_2025_capped, 
            dataset_names[0], 
            df_hdr_composite_2025_capped, 
            dataset_names[1]
        )

        print(f'---------- *\nSection 3.4: Merged {dataset_names[0]} and {dataset_names[1]} dataset information:')
        print(merged_step_03.info(), '\n')

        # Impute the missing null values as a result of merging the two datsets
        numeric_cols = merged_step_03.select_dtypes(include='number').columns
        merged_step_03[numeric_cols] = numeric_imputer.fit_transform(
            merged_step_03[numeric_cols]
            )
        categorical_cols = merged_step_03.select_dtypes(include='object').columns
        merged_step_03[categorical_cols] = categorical_imputer.fit_transform(
            merged_step_03[categorical_cols]
            )
        print(f'---------- *\nSection 3.4: Updated {dataset_names[0]} and {dataset_names[1]} dataset information:')
        print(merged_step_03.info(), '\n')
        return  merged_step_03