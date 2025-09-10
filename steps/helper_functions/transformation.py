import numpy as np 
import pandas as pd
from scipy import sparse
from imblearn.over_sampling import SMOTENC
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.pipeline import Pipeline

# ---------- * Helper fuctions * ----------
class transformation:
    ########
    # Function to address class imbalance with SMOTENC
    ########
    def apply_smote(self, dataframe, feature_name):
        # Split target / features
        y = dataframe[feature_name]
        X = dataframe.drop(columns=[feature_name]).copy()

        # Assert no NaNs (raise if found)
        if X.isna().any().any() or pd.isna(y).any():
            raise ValueError("Dataset contains NaNs but imputation is disabled. Please clean first.")

        # Identify categorical columns
        cat_cols = X.select_dtypes(exclude='number').columns.tolist()

        # Encode categoricals to integer codes (required by SMOTENC)
        for selected_feature in cat_cols:
            X[selected_feature] = pd.Series(X[selected_feature], dtype='category').cat.codes  

        # Indices of categorical columns (in current X order)
        cat_idx = [X.columns.get_loc(c) for c in cat_cols]

        # Class check
        if y.nunique() < 2:
            raise ValueError("SMOTENC needs at least two classes in y.")

        # Ensure y is a 1-D integer
        y = pd.Series(y).astype(int)

        # Choose k based on minority count (must be < minority_count)
        minority_count = y.value_counts().min()
        k = max(1, min(5, minority_count - 1))

        smotenc = SMOTENC(categorical_features=cat_idx, random_state=42, k_neighbors=k)
        result = smotenc.fit_resample(X.values, y.values)

        # imbalanced-learn may return 2- or 3-tuple (with sample weights)
        if isinstance(result, tuple) and len(result) == 3:
            X_resampled, y_resampled, _ = result
        else:
            X_resampled, y_resampled = result

        X_resampled = self.to_2d_frame(X_resampled, columns=X.columns)
        y_resampled = self.to_1d_series(y_resampled, name="own_mobile_phone").astype(int)
        return X, y, X_resampled, y_resampled

    ########
    # Function to impute half of a column’s missing values with a constant 
    # and the other half with the median value
    ########
    def to_2d_frame(self, X_like, columns=None) -> pd.DataFrame:
        if isinstance(X_like, pd.DataFrame):
            return X_like.reset_index(drop=True)
        arr = np.asarray(X_like)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return pd.DataFrame(arr, columns=columns)
    
    ########
    # Function to convert different input types into a clean 1-D pandas Series.
    ########
    def to_1d_series(self, y_like, name="target") -> pd.Series:
        # Already a Series → return a clean copy
        if isinstance(y_like, pd.Series):
            return y_like.reset_index(drop=True)

        # Single-column DataFrame → take that column
        if isinstance(y_like, pd.DataFrame):
            if y_like.shape[1] != 1:
                raise ValueError(f"{name} must be 1-D; got DataFrame with {y_like.shape[1]} columns.")
            return y_like.iloc[:, 0].reset_index(drop=True)

        # Anything else → make an array and squeeze/ravel to 1-D
        arr = np.asarray(y_like)
        if arr.ndim > 1:
            if arr.shape[1] == 1:
                arr = arr.ravel()
            else:
                raise ValueError(f"{name} must be 1-D; got array with shape {arr.shape}.")
        return pd.Series(arr, name=name)

    ########    
    # Function to rank features by Pearson correlation with y and select the most relevant.
    # Returns:
    #    X_selected   : DataFrame with selected columns
    #    corr_sorted  : Series of correlations (signed), sorted by |corr| desc
    #    retained     : list of kept column names
    ########    
    def select_by_correlation(
        self,
        X_resampled,
        y_resampled,
        top_k: int | None = 20,
        min_abs_corr: float | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        
        # Ensure DataFrame / Series
        if not isinstance(X_resampled, pd.DataFrame):
            X_resampled = pd.DataFrame(
                np.asarray(X_resampled),
                columns=[f"f{i}" for i in range(np.asarray(X_resampled).shape[1])]
            )
        if not isinstance(y_resampled, pd.Series):
            y_resampled = pd.Series(np.asarray(y_resampled), name="own_mobile_phone")

        # Align indices (just in case)
        X_resampled = X_resampled.reset_index(drop=True)
        y_resampled = y_resampled.reset_index(drop=True)

        # Compute Pearson correlations feature-by-feature
        corrs = {}
        for c in X_resampled.columns:
            s = pd.to_numeric(X_resampled[c], errors="coerce")
            if s.nunique(dropna=True) < 2:
                corrs[c] = np.nan
                continue
            corrs[c] = s.corr(y_resampled, method="pearson")

        corr_series = pd.Series(corrs, name="corr_with_target")
        corr_sorted = corr_series.reindex(
            corr_series.abs().sort_values(ascending=False).index
        )

        # Decide which features to keep
        retained = corr_sorted.index.tolist()
        if top_k is not None:
            retained = retained[:top_k]
        if min_abs_corr is not None:
            retained = [c for c in retained if abs(corr_series[c]) >= min_abs_corr]

        # Drop NaN correlations if any snuck in
        retained = [c for c in retained if pd.notna(corr_series[c])]
        X_selected = X_resampled[retained].copy()

        return X_selected, corr_sorted, retained
    
    ########    
    # Function to robustly retrieve feature names from a column transformer
    ########  
    def ct_feature_names(self, ct) -> List[str]:
            try:
                return list(ct.get_feature_names_out())
            except Exception:
                names = []
                for name, trans, cols in ct.transformers_:
                    if trans == 'drop':
                        continue
                    if trans == 'passthrough':
                        names.extend([str(c) for c in cols])
                        continue
                    # Handle Pipeline(last step) or direct transformer
                    if hasattr(trans, "named_steps"):
                        last = trans.named_steps.get("oh", trans.steps[-1][1])
                    else:
                        last = trans
                    if hasattr(last, "get_feature_names_out"):
                        try:
                            names.extend(list(last.get_feature_names_out(cols)))
                        except TypeError:
                            names.extend(list(last.get_feature_names_out()))
                    else:
                        names.extend([f"{name}__{c}" for c in cols])
                return names

    ########    
    # Function to normalise data
    ########  
    def normalise_data(self, dataframe):

        exclude_cols = ["year", "income_group_binary"]
        
        # Split numeric vs categorical BEFORE transforming
        num_cols = dataframe.select_dtypes(include=np.number).columns.tolist()
        cat_cols = dataframe.select_dtypes(exclude=np.number).columns.tolist()

        num_cols = [c for c in num_cols if c not in exclude_cols]

        # Map to N(0,1) via quantiles
        num_pipe = Pipeline([
            ('qt', QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)),
        ])

        cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        ct = ColumnTransformer(
            transformers=[
                ("", "passthrough", exclude_cols),
                ("num", num_pipe, num_cols),  
                ("cat", cat_pipe, cat_cols)
            ],
            remainder="drop",   
        )

        # Transform
        X_norm_np = ct.fit_transform(dataframe) 

        # Get feature names robustly
        feature_names: List[str] = self.ct_feature_names(ct)

        # Fix common mismatches
        n_rows, n_cols = X_norm_np.shape
        if len(feature_names) != n_cols:
            # Fallback to generic names
            feature_names = [f"f_{i}" for i in range(n_cols)]

        # Build a clean index 
        idx = dataframe.index if isinstance(dataframe, pd.DataFrame) else pd.RangeIndex(n_rows)
        if not isinstance(idx, pd.Index):
            idx = pd.Index(idx)

        if sparse.issparse(X_norm_np):
            X_norm_np = X_norm_np.A  # type: ignore[attr-defined]
        else:
            X_norm_np = np.asarray(X_norm_np)   

        # Build the DataFrame 
        X_norm = pd.DataFrame(X_norm_np, columns=feature_names)
        X_norm.index = idx
        return X_norm
# ----------END: * Helper fuctions * ----------
