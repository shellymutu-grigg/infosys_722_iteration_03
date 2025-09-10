import re
from datetime import datetime

# ---------- * Helper fuctions * ----------

class dataset:    
    ########
    # Function to merge two datsets on country code and year
    ########
    def merge_datasets(self, dataframe_01, dataframe_01_name, dataframe_02, dataframe_02_name):
        # --- 1) Identify HDR 'year-suffixed' columns and melt wide → long ---
        hdr_id_vars = ['code', 'country', 'region', 'human_development_groups_level']
        hdr_id_vars = [c for c in hdr_id_vars if c in dataframe_02.columns]

        hdr_year_cols = [c for c in dataframe_02.columns if re.search(r'_(\d{4})$', c)]

        hdr_long = dataframe_02.melt(
            id_vars=hdr_id_vars,
            value_vars=hdr_year_cols,
            var_name='indicator',
            value_name='value'
        )

        # Extract year and base metric name
        hdr_long['year'] = hdr_long['indicator'].str.extract(r'_(\d{4})$', expand=False).astype(int)
        hdr_long['metric'] = hdr_long['indicator'].str.replace(r'_(\d{4})$', '', regex=True)

        # --- 2) Pivot long → wide ---
        hdr_tidy = (
            hdr_long
            .pivot_table(index=['code', 'year'], columns='metric', values='value', aggfunc='first')
            .reset_index()
        )

        # --- 2b) Bring ONLY static cols we want from HDR ---
        wanted_static = ['code',
                        'human_development_groups_level_binary',
                        'human_development_groups_level_binary_label']
        wanted_static = [c for c in wanted_static if c in dataframe_02.columns]

        if wanted_static:
            static_cols = dataframe_02[wanted_static].drop_duplicates('code')
            hdr_tidy = hdr_tidy.merge(static_cols, on='code', how='left')

        # --- 3) Drop duplicate columns except protected ---
        protected_cols = {'human_development_groups_level_binary',
                        'human_development_groups_level_binary_label'}
        overlap = (set(hdr_tidy.columns) & set(dataframe_01.columns)) - {'code', 'year'} - protected_cols
        if overlap:
            hdr_tidy = hdr_tidy.drop(columns=list(overlap))

        # --- 4) Merge ---
        df_merged = dataframe_01.merge(hdr_tidy, on=['code', 'year'], how='left')

        # --- 4b) Drop unwanted columns ---
        drop_cols = [c for c in ['human_development_groups_level', 'income_group'] if c in df_merged.columns]
        if drop_cols:
            df_merged = df_merged.drop(columns=drop_cols)

        # --- 5) Save merge ---
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        out_path = f'{stamp}_merged_{dataframe_01_name}_with_{dataframe_02_name}.csv'
        df_merged.to_csv(out_path, index=False, encoding='utf-8', na_rep='')
        print(f'Saved merged dataset to: {out_path}')

        # Quick sanity check
        if 'human_development_groups_level' in df_merged.columns or 'income_group' in df_merged.columns:
            print('WARNING: Some unwanted columns still remain!')

        return df_merged

# ----------END: * Helper fuctions * ----------
