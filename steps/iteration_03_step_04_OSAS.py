# -*- coding: utf-8 -*-
from helper_functions.transformation import transformation as TR
from helper_functions.visualisations import visualisations as V


'''
Please see accompanying 'INFOSYS 722 Iteration 03 : STEPS 1 - 8 (OSAS)_(mmut001, 9564814)' document
for further documentation of this code

NOTE: If running this in VSCode, please ensure the Figure files generated to visulaize the data are 
closed as they appear to permit the code to continue to run

************************************************************************

THE CODE BELOW ONLY DEALS WITH STEP 04 OF THE DATA MINING PROCESS (OSAS) 

************************************************************************

'''

tr = TR()
v = V()
FACTOR = 3.0

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
Please see Section 3 in the accompanying report & code in 'iteration_03_step_03_OSAS' for further details
'''

# 04-DT
'''
Complete data transformation step
'''

class step_04:
    def data_transformation(self, dataframe):
        # ---------- * Section 4.1 : Data Reduction in accompanying report * ----------
        merged_step_04 = dataframe.drop(columns=['gender'])

        print(f'---------- *\nSection 4.1: Updated merged dataset information:')
        print(merged_step_04.info(), '\n')

        X, y,  X_resampled, y_resampled = tr.apply_smote(merged_step_04, 'own_mobile_phone')

        print('---------- *\nSection 4.1: X features shape BEFORE SMOTE oversampling:')
        print(X.shape, y.value_counts(dropna=False).to_dict(), '\n')

        print('---------- *\nSection 4.1: X features shape AFTER SMOTE oversampling:')
        print(X_resampled.shape, y_resampled.value_counts(dropna=False).to_dict(), '\n')

        # Conduct horizontal feature reduction and keep only the most correlated features with own_mobile_phone
        # using Pearson method
        X_resampled, corr_rankings, retained = tr.select_by_correlation(
            X_resampled,
            y_resampled,
            top_k = 30,           
            min_abs_corr = 0.0005   
        )
        # Sort correlations by absolute value (desc) and show a snapshot
        corr_sorted = corr_rankings.dropna().reindex(
            corr_rankings.abs().sort_values(ascending=False).index
        )

        print('---------- *\nSection 4.1: Features correlation rankings:')
        for idx, (feature, corr_val) in enumerate(corr_sorted.head(min(len(corr_sorted), len(retained))).items(), start=1):
            print(f'{idx:2d}. {feature:40s}\t\t\t\t{corr_val:.6f}')
        print()

        print('---------- *\nSection 4.1: Features retained after horizontal feature reduction:')
        for idx, feature in enumerate(retained, start=1):
            print(f'{idx:>2}. {feature}')
        print()

        print('---------- *\nSection 4.1: X_resampled features shape AFTER horizontal feature reduction:')
        print(X_resampled.shape, y_resampled.value_counts(dropna=False).to_dict(), '\n')

        # ---------- * Section 4.2 : Data Projection in accompanying report * ----------
        v.plot_feature_distributions(X_resampled, y_resampled, 'own_mobile_phone')

        X_norm = tr.normalise_data(X_resampled)

        # Quick checks (for numeric block)
        means = X_norm.mean(numeric_only=True)
        stds  = X_norm.std(numeric_only=True)
        print('---------- *\nSection 4.1: X_norm numeric means (should be ~0):')
        print(means.head(10), '\n')
        print('---------- *\nSection 4.1: X_norm numeric stds (should be ~1):')
        print(stds.head(10), '\n')

        print('---------- *\nSection 4.2: X_norm features shape AFTER normal distribution Transformation:')
        print(f'{X_norm.shape} {y_resampled.value_counts(dropna=False).to_dict()}', '\n')

        v.plot_feature_distributions(X_norm, y_resampled, 'own_mobile_phone')

        return X_norm, y_resampled

