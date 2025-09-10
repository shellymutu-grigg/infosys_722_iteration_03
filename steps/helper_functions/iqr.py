import numpy as np 
import pandas as pd
from datetime import datetime

# ---------- * Helper fuctions * ----------
class iqr:

    ########
    # Function to determine Column-wise IQR bounds for all numeric columns in dataframe
    ########
    def iqr_bounds_all(self, df, factor) -> pd.DataFrame:
        num = df.select_dtypes(include='number')
        if num.empty:
            return pd.DataFrame(columns=['lower', 'upper'])
        q1 = num.quantile(0.25)
        q3 = num.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return pd.DataFrame({'lower': lower, 'upper': upper})

    ########
    # Function to determine IQR bounds for a single numeric Series; returns (lower, upper) floats.
    ########
    def iqr_bounds_series(self, s: pd.Series, factor) -> tuple[float, float]:
        s = pd.to_numeric(s, errors='coerce').dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = float(q1 - factor * iqr)
        upper = float(q3 + factor * iqr)
        return lower, upper

    ########
    # Handle outliers in numeric columns using the IQR method
    #   strategy:
    #       - "cap": Winsorize values to within IQR bounds
    #   exclude: list of numeric columns to skip (e.g., IDs, binary flags)
    ########
    def handle_outliers_iqr(
        self, 
        dataframe: pd.DataFrame, 
        factor: float, 
        strategy: str = "cap", 
        exclude: list[str] | None = None
    ) -> pd.DataFrame:
        
        capped_dataframe = dataframe.copy()

        # Calculate column-wise IQR bounds (returns DataFrame with lower/upper)
        bounds_summary_dataframe = self.iqr_bounds_all(capped_dataframe, factor=factor)

        # Determine numeric columns to check
        numeric_columns = capped_dataframe.select_dtypes(include="number").columns.tolist()
        if exclude:
            numeric_columns = [col for col in numeric_columns if col not in set(exclude)]

        # Extract matching lower and upper bounds for selected columns
        lower_bounds = bounds_summary_dataframe.loc[numeric_columns, "lower"]
        upper_bounds = bounds_summary_dataframe.loc[numeric_columns, "upper"]

        if strategy == "cap":
            # Winsorize each numeric column
            for col in numeric_columns:
                capped_dataframe[col] = np.clip(capped_dataframe[col], lower_bounds[col], upper_bounds[col])
        else:
            raise ValueError("Invalid strategy. Must be 'cap' or 'remove'.")

        return capped_dataframe

    ########
    # Function to detect outliers column-by-column using IQR in a specified dataframe
    # and export summary + flagged rows to CSV
    # Returns: pct_outlier_rows
    ########
    def find_outliers_iqr(
        self, 
        dataframe: pd.DataFrame,
        dataframe_name: str,
        factor: float,
        exclude: list[str] | None = None,
        min_unique: int = 5,
        filename_prefix: str = "outliers_report"
    ) :
        exclude_set = set(exclude or [])
        numeric_cols = [c for c in dataframe.select_dtypes(include="number").columns if c not in exclude_set]

        per_col = {}
        union_mask = pd.Series(False, index=dataframe.index)

        rows = []
        outlier_rows = []  # collect per-column outliers

        for col in numeric_cols:
            series = pd.to_numeric(dataframe[col], errors="coerce")
            if series.nunique(dropna=True) < min_unique:    # skip near-constant/binary columns
                continue
            lo, hi = self.iqr_bounds_series(series, factor)
            mask = (series < lo) | (series > hi)

            per_col[col] = {
                "lower": lo,
                "upper": hi,
                "valid_records": int(series.notna().sum()),
                "num_outliers": int(mask.sum()),
                "mask": mask
            }

            if mask.sum() > 0:
                rows.append({
                    "column": col,
                    "lower": lo,
                    "upper": hi,
                    "valid_records": int(series.notna().sum()),
                    "num_outliers": int(mask.sum()),
                    "pct_outliers": (mask.sum() / max(1, series.notna().sum())) * 100.0
                })

                # collect the actual outlier rows for this column
                temp_df = dataframe.loc[mask].copy()
                temp_df.insert(0, "outlier_column", col)  # insert the column name at the first position
                outlier_rows.append(temp_df)

            union_mask |= mask.fillna(False)

        # Build tidy report DataFrame
        report_df = pd.DataFrame(rows).sort_values("pct_outliers", ascending=False)

        # Export report
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_path = f"{dataframe_name}_{filename_prefix}_summary_{stamp}.csv"
        report_df.to_csv(report_path, index=False, encoding="utf-8")

        # Export actual outlier rows (stacked for all columns)
        if outlier_rows:
            outliers_df = pd.concat(outlier_rows, ignore_index=True)
            outliers_path = f'{dataframe_name}_{filename_prefix}_rows_{stamp}.csv'
            outliers_df.to_csv(outliers_path, index=False, encoding="utf-8")
            print(f'Exported summary of outliers in {dataframe_name} to: {report_path}')
            print(f'Exported the actual outlier rows in {dataframe_name} to: {outliers_path}')
        else:
            outliers_path = ""   # return empty string if no file exported
            print(f"No outliers detected in {dataframe_name}, no rows exported.")

        # Calculate percentage of rows with ≥1 outlier
        pct_outlier_rows = float(union_mask.mean() * 100)

        print(f"Percentage of rows with at least one outlier in {dataframe_name}: {pct_outlier_rows:.2f}%")

    ########
    # Function to impute half of a column’s missing values with a constant 
    # and the other half with the median value
    ########
    def impute_missing_values(self, dataframe, column, constant, random_state=42):
        np.random.seed(random_state)

        # indices of missing values
        nan_idx = dataframe.loc[dataframe[column].isna()].index

        if len(nan_idx) == 0:
            # nothing to impute
            return dataframe

        # shuffle so it's random which NaNs get median vs constant
        nan_idx = np.random.permutation(nan_idx)

        # split into two halves
        half = len(nan_idx) // 2
        idx_median = nan_idx[:half]
        idx_const = nan_idx[half:]

        # fill with median
        median_val = dataframe[column].median()
        dataframe.loc[idx_median, column] = median_val

        # fill with constant
        dataframe.loc[idx_const, column] = constant

        return dataframe

# ----------END: * Helper fuctions * ----------
