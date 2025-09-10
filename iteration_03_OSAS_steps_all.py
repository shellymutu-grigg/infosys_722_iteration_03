from steps.iteration_03_step_02_OSAS import step_02
from steps.iteration_03_step_03_OSAS import step_03
from steps.iteration_03_step_04_OSAS import step_04

"""
Please see accompanying 'INFOSYS 722 Iteration 03 : STEPS 1 - 8 (OSAS)_(mmut001, 9564814)' document
for further documentation of this code

NOTE: If running this in VSCode, please ensure the Figure files generated to visulaize the data are 
closed as they appear to permit the code to continue to run

*******************************************************************************

THE CODE BELOW IS THE EXECUTION OF ALL OF THE DATA MINING PROCESS STEPS (OSAS) 

*******************************************************************************

"""

FACTOR = 3.0

s2 = step_02()
s3 = step_03()
s4 = step_04()

datasets_s2 = s2.data_understanding(
    'GlobalFindexDatabase2025.csv', 
    'HDR25_Composite_indices_complete_time_series.csv', 
    'world-regions-according-to-the-world-bank.csv'
    )
df_g_f_db_2025_s2          = datasets_s2[0]
df_hdr_composite_2025_s2   = datasets_s2[1]

datasets_s3 = s3.data_preparation(df_g_f_db_2025_s2, df_hdr_composite_2025_s2, FACTOR)
merged_s3          = datasets_s3

datasets_s4 = s4.data_transformation(merged_s3)
df_g_f_db_2025_s4          = datasets_s4