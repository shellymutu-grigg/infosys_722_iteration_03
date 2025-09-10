import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
from math import ceil

# ---------- * Helper fuctions * ----------
class visualisations:
    ########
    # Function to determine IQR bounds for a single numeric Series; returns (lower, upper) floats.
    ########
    def iqr_bounds_series(self, series: pd.Series, factor) -> tuple[float, float]:
        series = pd.to_numeric(series, errors='coerce').dropna()
        quantile1 = series.quantile(0.25)
        quantile3 = series.quantile(0.75)
        iqr = quantile3 - quantile1
        lower = float(quantile1 - factor * iqr)
        upper = float(quantile3 + factor * iqr)
        return lower, upper
    
    ########
    # Function to display horizontal histogram for a binary/categorical column
    ########
    def show_horizontal_histogram(self, dataframe, column, dataframe_name, labels=None):
        if column not in dataframe.columns:
            print(f"Column '{column}' not found in {dataframe_name}")
            return
        
        # Count values
        counts = dataframe[column].value_counts().sort_index()
        
        # Default labels (0 and 1 → strings)
        if labels is None:
            labels = {0: "0", 1: "1"}
        
        # Colors: assign one per category
        colors = ['#6395EE', '#E76F51', '#2A9D8F', '#F4A261', '#E9C46A']
        
        # Plot horizontal bar chart
        plt.figure(figsize=(8, 4))
        counts.plot(
            kind='barh',
            color=colors[:len(counts)],
            edgecolor='black'
        )
        
        # Force all labels to str so matplotlib is happy
        ytick_labels = [str(labels.get(i, i)) for i in counts.index]
        plt.yticks(ticks=range(len(counts)), labels=ytick_labels)
        
        plt.title(f'{column.capitalize()} distribution in {dataframe_name} Dataset', fontsize=14)
        plt.xlabel('# of instances', fontsize=12)
        plt.ylabel(column, fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show(block=True)

    ########
    # Function to display Dot Chart
    ########  
    def show_dot_chart(self, dataframe, x, y, hue):
        plt.figure(figsize=(10, 6))
        sns.stripplot(
            data=dataframe,
            x=x, 
            y=y,
            hue=hue,          # color by gender
            jitter=True, 
            dodge=True,
            size=4
        )

        # Title with extra padding
        plt.title(f'Dot Chart: {hue} & {x} vs. {y}', fontsize=14, pad=25)

        # Axis labels with padding
        plt.xlabel(f'{x}', fontsize=12, labelpad=15)
        plt.ylabel(f'{y}', fontsize=12, labelpad=15)

        # Add spacing for tick labels
        plt.tick_params(axis='x', which='major', pad=10, labelsize=10)
        plt.tick_params(axis='y', which='major', pad=10, labelsize=10)

        # Add faded grid
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

        plt.legend(title=f'{hue}')
        plt.tight_layout()
        plt.show()

    ########
    # Function to display boxplots of dataframe features with outliers
    ########
    def show_boxplots_with_outliers(self, dataframe, dataframe_name, factor, max_columns=5):
        numeric_cols = dataframe.select_dtypes(include='number').columns
        cols_with_outliers = []

        # Detect outliers column by column
        for col in numeric_cols:
            series = dataframe[col].dropna()
            if series.nunique() < 5:
                continue 
            lo, hi =self.iqr_bounds_series(series, factor)
            if ((series < lo) | (series > hi)).any():
                cols_with_outliers.append(col)
        
        if not cols_with_outliers:
            print(f'{dataframe_name}: No outliers detected by IQR.\n')
            return []

        # Setup subplot grid
        num_cols = max_columns
        num_rows = ceil(len(cols_with_outliers) / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.6, num_rows * 3.2))
        axes = np.atleast_1d(axes).ravel()

        for index, column in enumerate(cols_with_outliers):
            series = pd.to_numeric(dataframe[column], errors="coerce").dropna()
            lo, hi = self.iqr_bounds_series(series, factor)
            axes[index].boxplot(series, vert=True, showfliers=True)
            axes[index].axhline(lo, linestyle="--", linewidth=1)
            axes[index].axhline(hi, linestyle="--", linewidth=1)
            axes[index].set_title(column, fontsize=9)
            axes[index].set_xticks([])  # cleaner
            axes[index].grid(True, axis='y', alpha=0.12)

        # hide unused panes
        for j in range(index + 1, len(axes)):
            axes[j].set_visible(False)

        # Title and spacing
        fig.suptitle(f'Boxplots of {dataframe_name} (IQR outliers, factor={factor})', fontsize=16, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        plt.subplots_adjust(hspace=0.3)
        plt.show()
        return cols_with_outliers

    ########
    # Function to display histograms of dataframe features with outliers
    ########
    def show_all_histograms_with_outliers(
        self, 
        df: pd.DataFrame,
        df_name: str,
        factor: float,
        bins: int = 40,
        max_cols: int = 5,
        min_unique: int = 5,
    ):
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # find columns that actually have outliers
        features_with_outliers = []
        for feature in numeric_cols:
            series = pd.to_numeric(df[feature], errors="coerce").dropna()
            if series.nunique() < min_unique:
                continue
            lo, hi = self.iqr_bounds_series(series, factor=factor)
            if ((series < lo) | (series > hi)).any():
                features_with_outliers.append(feature)

        if not features_with_outliers:
            print("No numeric columns with outliers found by IQR.")
            return []

        # layout
        ncols = max(1, int(max_cols))
        nrows = int(np.ceil(len(features_with_outliers) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.6))
        axes = np.atleast_1d(axes).ravel()

        # plot each
        for index, feature in enumerate(features_with_outliers):
            ax = axes[index]
            series = pd.to_numeric(df[feature], errors="coerce").dropna()
            lo, hi = self.iqr_bounds_series(series, factor=factor)

            # inside/outside split
            mask_out = (series < lo) | (series > hi)
            inside = series[~mask_out]
            outliers = series[mask_out]

            ax.hist(inside, bins=bins, alpha=0.9, edgecolor="white")
            if len(outliers):
                # jittered rug along baseline
                y0 = np.full(outliers.shape[0], ax.get_ylim()[0] + 1e-6)
                jitter = (np.random.rand(outliers.shape[0]) - 0.5) * 0.002 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.scatter(outliers.values, y0 + jitter, s=10, marker="o", alpha=0.8)

            # IQR bounds
            ax.axvline(lo, linestyle="--", linewidth=1)
            ax.axvline(hi, linestyle="--", linewidth=1)

            ax.set_title(feature, fontsize=9)
            ax.set_xlabel("")   # keep clean; avoids clutter
            ax.set_ylabel("# of instances", fontsize=8)
            ax.grid(True, axis="y", alpha=0.15)
            ax.text(0.98, 0.95, f"n={len(series)} | out={mask_out.sum()}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=8)

        # hide any unused axes
        for j in range(index + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f'{df_name} histograms for features with IQR Outliers (factor={factor})', fontsize=14, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show()

        return features_with_outliers

    ########
    # Function to print outlier feastures to the command line
    ########
    def print_outlier_columns(self, outliers_list):
        for feature in outliers_list:
            print(f"  - {feature}")
        print()

    ########
    # Function to plot histograms for each feature in X, plus the target y
    ########
    def plot_feature_distributions(self, X: pd.DataFrame, y: pd.Series, feature_name, max_cols: int = 7, bins: int = 30):
        # Ensure y is a Series with a name
        y = pd.Series(y).reset_index(drop=True)
        y.name = feature_name           
        y_col: str = str(y.name)              
        y_df = y.to_frame(name=y_col)

        n_features = X.shape[1] + 1  # features + target
        ncols = max_cols
        nrows = int(np.ceil(n_features / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = np.atleast_1d(axes).ravel()

        # Plot each feature (pass via data=..., x=...)
        for index, feature in enumerate(X.columns):
            ax = axes[index]
            sns.histplot(data=X, x=feature, bins=bins, kde=False, ax=ax, color="skyblue", edgecolor="black")
            ax.set_title(f"{feature}", fontsize=9)
            ax.set_xlabel("")
            ax.grid(True, linestyle="--", alpha=0.5)
            # Rotate x-axis labels for readability
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment("right")

        # Plot target
        ax = axes[len(X.columns)]
        y_df = y.to_frame(name=y.name)
        sns.histplot(data=y_df, x=y.name, bins=max(len(y.unique()), 2), discrete=True,
                    ax=ax, color="salmon", edgecolor="black")
        ax.set_title(f'{feature_name} (target)', fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Hide any unused subplots
        for j in range(len(X.columns) + 1, len(axes)):
            axes[j].set_visible(False)

        # Title for the entire figure
        fig.suptitle("Retained Features Distributions", fontsize=16)

        # Adjust layout so title is visible
        plt.tight_layout(rect=(0, 0, 1, 0.96))  # leave space at top for suptitle
        plt.show()     

# ----------END: * Helper fuctions * ----------
