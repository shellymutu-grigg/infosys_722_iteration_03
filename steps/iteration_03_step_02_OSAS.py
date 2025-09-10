# -*- coding: utf-8 -*-
import pandas as pd
from helper_functions.data_cleaning import data_cleaning
from helper_functions.visualisations import visualisations

'''
Please see accompanying 'INFOSYS 722 Iteration 03 : STEPS 1 - 8 (OSAS)_(mmut001, 9564814)' document
for further documentation of this code

NOTE: If running this in VSCode, please ensure the Figure files generated to visulaize the data are 
closed as they appear to permit the code to continue to run

************************************************************************

THE CODE BELOW ONLY DEALS WITH STEP 02 OF THE DATA MINING PROCESS (OSAS) 

************************************************************************

'''
dc = data_cleaning()
v = visualisations()
dataset_names = ['GlobalFindexDatabase2025', 'HDR25_Composite_indices_complete_time_series', 'world-regions-according-to-the-world-bank']

# 01-BU

'''
Please see Section 1 in the accompanying report for further details of the business understanding
'''


# 02-DU
'''
Complete data understanding steps to gain a deeper understanding of the data being worked with

NOTE: some transformations cover Section 3 areas as well and are marked as such
'''

# ---------- * Section 2.3: Data Exploration in accompanying report * ----------

class step_02:
    def data_understanding(self, dataframe_fin_name, dataframe_hdr_name, dataframe_regions_name):
        # Explore Data
        # Load Datasets and ensure data types are as expected
        g_f_db_2025_csv = dataframe_fin_name
        df_g_f_db_2025 = pd.read_csv('datasets/'+g_f_db_2025_csv, dtype={
            'countrynewwb': str, 'codewb': str, 'year': int, 'pop_adult': float, 'regionwb24_hi': str, 'incomegroupwb24': str, 'group2': str, 'account.t.d': float, 
            'fiaccount.t.d': float, 'mobileaccount.t.d': float, 'borrow.any.td': float, 'fin17a.17a1.d': float, 'fin22b': float, 'fin24aP': float, 'fin24aN': float, 
            'fin30': float, 'fin32.n33': float, 'fin32': float, 'fin32.acc': float, 'fin37.38': float, 'fin2.t.d': float, 'fin10': float, 'fing2p': float, 
            'g20.made': float, 'g20.received': float, 'g20.any': float, 'save.any.t.d': float, 'con1': float
            })

        hdr_composite_csv = dataframe_hdr_name
        df_hdr_composite_2025 = pd.read_csv('datasets/'+hdr_composite_csv, dtype={
            'iso3': str, 'country': str, 'hdicode': str, 'region': str, 'gdi_2011': float, 'gdi_2014': float, 'gdi_2017': float, 'gdi_2021': float, 
            'hdi_f_2011': float, 'hdi_f_2014': float, 'hdi_f_2017': float, 'hdi_f_2021': float, 'mys_f_2011': float, 'mys_f_2014': float, 
            'mys_f_2017': float, 'mys_f_2021': float, 'gni_pc_f_2011': float, 'gni_pc_f_2014': float, 'gni_pc_f_2017': float, 'gni_pc_f_2021': float,
            'hdi_m_2011': float, 'hdi_m_2014': float, 'hdi_m_2017': float,  'hdi_m_2021': float, 'eys_m_2011': float, 'eys_m_2014': float, 'eys_m_2017': float, 
            'eys_m_2021': float, 'mys_m_2011': float, 'mys_m_2014': float, 'mys_m_2017': float, 'mys_m_2021': float, 'gni_pc_m_2011': float, 'gni_pc_m_2014': float, 
            'gni_pc_m_2017': float, 'gni_pc_m_2021': float, 'ineq_edu_2011': float, 'ineq_edu_2014': float, 'ineq_edu_2017': float, 'ineq_edu_2021': float, 
            'ineq_inc_2011': float, 'ineq_inc_2014': float, 'ineq_inc_2017': float, 'ineq_inc_2021': float
            })

        world_regions_2023_csv = dataframe_regions_name
        df_world_regions_2023 = pd.read_csv('datasets/'+world_regions_2023_csv, dtype={ 'Entity': str, 'Code': str, 'Year': int, 'World regions according to WB': str })

        print(f'---------- *\nSection 2.3: {dataset_names[0]} dataset information:')
        print(df_g_f_db_2025.info(), '\n')

        print(f'---------- *\nSection 2.3: {dataset_names[1]} dataset information:')
        print(df_hdr_composite_2025.info(), '\n')

        print(f'---------- *\nSection 2.3: {dataset_names[2]} dataset information:')
        print(df_world_regions_2023.info(), '\n')

        # ---------- * Section 2.3: Data Exploration in accompanying report * ----------
        # ---------- * Section 2.4: Verify the Data Quality in accompanying report * ----------
        # ---------- * Section 3.2: Data Cleaning in accompanying report * ----------
        # Rename the country features across all datasets to allow merging and lookup of datasets
        df_g_f_db_2025.rename(columns={'countrynewwb': 'country', 'codewb': 'code'}, inplace=True)
        df_world_regions_2023.rename(columns={'Entity': 'country', 'Code': 'code', 'Year': 'year', 'World regions according to WB': 'region'}, inplace=True)
        df_hdr_composite_2025.rename(columns={'iso3': 'code'}, inplace=True)

        # Rename the GlobalFindexDatabase2025 features for readability
        df_g_f_db_2025.rename(columns={
            'pop_adult': 'population', 'regionwb24_hi': 'region', 'incomegroupwb24': 'income_group', 'group2': 'gender', 'account.t.d': 'account',
            'fiaccount.t.d': 'bank_account', 'mobileaccount.t.d': 'mobile_account', 'borrow.any.t.d': 'borrow_01', 
            'fin17a.17a1.d': 'saved_money_at_bank', 'fin22b': 'borrow_02', 'fin24aP': 'emergency_01', 'fin24aN': 'emergency_02', 
            'fin30': 'utility', 'fin32.n33': 'wages_01', 'fin32': 'wages_02','fin32.acc': 'wages_03', 'fin37.38': 'pension', 
            'fin2.t.d': 'debit_card', 'fin10': 'credit_card', 'fing2p': 'government_pay', 'g20.made': 'made_digital_payment', 
            'g20.received': 'received_digital_payment', 'g20.any': 'any_digital_payment', 'save.any.t.d': 'save_any_money', 
            'con1': 'own_mobile_phone'
            }, inplace=True)

        # Remove the GlobalFindexDatabase2025 dataframe features that are not under analysis
        df_g_f_db_2025_columns = [
            'country', 'code', 'year', 'population', 'region', 'income_group', 'gender', 'account','bank_account', 'mobile_account', 
            'borrow_01','saved_money_at_bank', 'borrow_02', 'emergency_01', 'emergency_02', 'utility', 'wages_01', 'wages_02', 'wages_03', 
            'pension', 'debit_card', 'credit_card', 'government_pay', 'made_digital_payment', 'received_digital_payment', 'any_digital_payment', 
            'save_any_money', 'own_mobile_phone'
            ]
        df_g_f_db_2025.drop(columns=[col for col in df_g_f_db_2025.columns if col not in df_g_f_db_2025_columns], axis = 1, inplace = True)

        print(f'---------- *\nSection 2.4/ Section 3.2.1: Updated {dataset_names[0]} dataset information')
        print(df_g_f_db_2025.info(), '\n')

        # Remove records from the GlobalFindexDatabase2025 dataframe where country names are blank as associated codes 
        # do not match any known country code
        df_g_f_db_2025 = df_g_f_db_2025.dropna(subset=['country'])

        # Remove records from the GlobalFindexDatabase2025 dataframe where country codes do not align with 
        # HDR25_Composite_indices_complete_time_series
        remove_codes = ['XKX', 'TWN', 'PRI', 'SAS']
        df_g_f_db_2025 = df_g_f_db_2025[~df_g_f_db_2025['code'].isin(remove_codes)]
        print(f'---------- *\nSection 2.3: {dataset_names[0]} dataset information AFTER country name and code clean up:')
        print(df_g_f_db_2025.info(), '\n')


        # Remove records from the GlobalFindexDatabase2025 dataframe that are not 'men' or 'women' gender records
        selected_cols = ['gender']
        print(f'---------- *\nSection 2.3: {dataset_names[0]} dataset information BEFORE gender filter:')
        print(df_g_f_db_2025[selected_cols].info(), '\n')
        genders = ['men', 'women']
        df_g_f_db_2025 = df_g_f_db_2025[df_g_f_db_2025['gender'].isin(genders)]
        print(f'---------- *\nSection 2.3: {dataset_names[0]} dataset information AFTER gender filter:')
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

        print(f'---------- *\nSection 2.3: Updated {dataset_names[0]} dataset information')
        print(df_g_f_db_2025.info(), '\n')

        # Update GlobalFindexDatabase2025 dataframe to use world-regions-according-to-the-world-bank 
        # country names
        df_g_f_db_2025 = dc.update_country_names(df_world_regions_2023, df_g_f_db_2025)
        print(f'---------- *\nSection 2.3/ Section 3.2.2: Updated {dataset_names[0]} dataset information')
        print(df_g_f_db_2025['country'].describe())
        print(df_g_f_db_2025['country'].unique()[:20], '\n')

        # Replace GlobalFindexDatabase2025 dataset region feature to use region specified in world-regions-according-to-the-world-bank. 
        # Resolves records that have 'High income' in the region field
        df_g_f_db_2025 = dc.update_regions(df_world_regions_2023, df_g_f_db_2025)

        # Check if any 'High income' values remain
        count_high_income = (df_g_f_db_2025['region'].str.strip() == 'High income').sum()

        print(f'---------- *\nSection 2.4.1: {dataset_names[0]} dataset information AFTER updating regions:')
        if count_high_income == 0:
            print("No 'High income' values remain in the 'region' column.\n")
        else:
            print(f"Found {count_high_income} rows with 'High income' in the 'region' column.\n")

        print(f'---------- *\nSection 2.4.1: Updated {dataset_names[0]} dataset information:')
        print(df_g_f_db_2025.info(), '\n')

        # Rename the HDR25_Composite_indices_complete_time_series features for readability
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
            'code', 'country', 'human_development_groups_level', 'region', 'gender_development_index_2011', 'gender_development_index_2014', 
            'gender_development_index_2017', 'gender_development_index_2021', 'gender_development_index_2022', 'human_development_index_female_2011', 
            'human_development_index_female_2014', 'human_development_index_female_2017', 'human_development_index_female_2021', 
            'human_development_index_female_2022','mean_years_schooling_female_2011', 'mean_years_schooling_female_2014', 'mean_years_schooling_female_2017', 
            'mean_years_schooling_female_2021', 'mean_years_schooling_female_2022', 'gross_national_income_per_capita_female_2011', 
            'gross_national_income_per_capita_female_2014', 'gross_national_income_per_capita_female_2017', 
            'gross_national_income_per_capita_female_2021', 'gross_national_income_per_capita_female_2022', 'human_development_index_male_2011', 
            'human_development_index_male_2014', 'human_development_index_male_2017',  'human_development_index_male_2021', 'human_development_index_male_2022', 
            'mean_years_schooling_male_2011', 'mean_years_schooling_male_2014', 'mean_years_schooling_male_2017', 'mean_years_schooling_male_2021', 
            'mean_years_schooling_male_2022', 'gross_national_income_per_capita_male_2011', 'gross_national_income_per_capita_male_2014', 
            'gross_national_income_per_capita_male_2017', 'gross_national_income_per_capita_male_2021', 'gross_national_income_per_capita_male_2022', 
            'inequality_in_education_2011', 'inequality_in_education_2014', 'inequality_in_education_2017', 'inequality_in_education_2021', 
            'inequality_in_education_2022', 'inequality_in_income_2011', 'inequality_in_income_2014', 'inequality_in_income_2017', 'inequality_in_income_2021', 
            'inequality_in_income_2022']
        df_hdr_composite_2025.drop(columns=[col for col in df_hdr_composite_2025.columns if col not in df_hdr_composite_2025_columns], axis = 1, inplace = True)

        print(f'---------- *\nSection 2.4/ Section 3.2.1: Updated {dataset_names[1]} dataset information')
        print(df_hdr_composite_2025.info(), '\n')

        # Update HDR25_Composite_indices_complete_time_series dataframe records to use same regions 
        # as GlobalFindexDatabase2025 dataframe
        df_hdr_composite_2025 = dc.update_regions(df_world_regions_2023, df_hdr_composite_2025)
        print(f'---------- *\nSection 2.4.1/ Section 3.2.2: Updated {dataset_names[1]} dataset information')
        print(df_hdr_composite_2025.info(), '\n')

        # Update HDR25_Composite_indices_complete_time_series dataframe to use world-regions-according-to-the-world-bank 
        # country names
        df_hdr_composite_2025 = dc.update_country_names(df_world_regions_2023, df_hdr_composite_2025)
        print(f'---------- *\nSection 2.4/ Section 3.2.2:: Updated {dataset_names[1]} dataset information')
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

        print(f'---------- *\nSection 2.4/ Section 3.3.2/ Section 3.5: Updated {dataset_names[1]} dataset information')
        print(df_hdr_composite_2025.info(), '\n')

        return  df_g_f_db_2025, df_hdr_composite_2025