
# ---------- * Helper fuctions * ----------
class data_cleaning:
    ########
    # Function to update dataset records to use correct region
    ########
    def update_regions(self, region_dataframe, dataframe):
        # Keep only necessary columns from world regions
        wr_region_lookup = region_dataframe[['code', 'region']].drop_duplicates()

        # Merge with HDR dataset to bring in the correct region
        dataframe = dataframe.merge(
            wr_region_lookup,
            on='code',
            how='left',
            suffixes=('', '_wr')   # to avoid overwriting immediately
        )

        # Replace region values with those from world regions when available
        dataframe['region'] = dataframe['region_wr'].combine_first(dataframe['region'])

        # Drop the helper column
        dataframe.drop(columns=['region_wr'], inplace=True)
        return dataframe

    ########
    # Function to update dataset records to use correct country name
    ########
    def update_country_names(self, names_dataframe, dataframe):
        # --- Build lookup dictionary from world regions dataset ---
        code_to_country = names_dataframe.set_index('code')['country'].to_dict()

        # --- Update 'country' column in dataframe_02 ---
        dataframe['country'] = dataframe['code'].map(code_to_country).fillna(dataframe['country'])
        return dataframe

# ----------END: * Helper fuctions * ----------
