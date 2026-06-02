import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from src import config


def save_figure(filename: str, save_dir: str = None) -> None:
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
        print(f">> Saved: {filename}")


def calculate_iqr_limits(df: pd.DataFrame, column: str) -> tuple:
    """Calculate IQR limits for outlier detection."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr
    return q1, q3, iqr, upper_limit, lower_limit


def view_outliers(df: pd.DataFrame) -> None:
    """Print the count and percentage of IQR outliers for numeric columns."""
    numeric_columns = df.select_dtypes(include=['number']).columns
    total_dataset = df.shape[0]

    for column in numeric_columns:
        _, _, _, upper_limit, lower_limit = calculate_iqr_limits(df, column)
        df_outliers = df[(df[column] >= upper_limit) | (df[column] <= lower_limit)]
        total_outliers = df_outliers.shape[0]
        percentage_outliers = total_outliers / total_dataset

        if 0 < percentage_outliers < 1:
            print(f'[ {column} ] >> {total_outliers} outliers :: {percentage_outliers * 100:.2f}%')


def fix_data(df: pd.DataFrame, column: str, factor_limit: float) -> None:
    """Cap negative and high outlier values in place using a custom IQR factor."""
    _, q3, iqr, _, _ = calculate_iqr_limits(df, column)
    superior_limit = q3 + factor_limit * iqr

    print(f'>> {column} column adjustment')
    print(f'>> Q3: {q3:.2f}')
    print(f'>> IQR: {iqr:.2f}')
    print(f'>> Superior Limit (custom): {superior_limit:.2f}\n')

    df.loc[df[column] < 0, column] = 0
    df.loc[df[column] > superior_limit, column] = superior_limit
    print(df[column].describe().to_string())
    print()


def treat_outliers(df: pd.DataFrame, columns: list = None, factor_limit: float = 4) -> pd.DataFrame:
    """Apply notebook outlier treatment to the configured numeric columns."""
    df = df.copy()
    columns = columns or config.OUTLIER_COLUMNS
    print('>> contagem de outliers / outliers count \n')
    view_outliers(df)
    print('\n>> tratamento de outliers / outliers treatment \n')
    for column in columns:
        fix_data(df, column=column, factor_limit=factor_limit)
    return df


def plot_initial_eda(df: pd.DataFrame, save_dir: str = None) -> None:
    """Plot the notebook's first EDA charts."""
    save_dir = save_dir or config.OUTPUT_DIR
    day_order = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(ax=axes[0], x=df['day_name'], y=df['fare_amount'], hue=df['VendorID'], order=day_order, showfliers=False)
    axes[0].set_xlabel('day of week', fontsize=9, weight='bold')
    axes[0].set_ylabel('fare amount', fontsize=9, weight='bold')
    axes[0].set_title('Graph 1 - Fare amount x day of week', fontsize=12)

    sns.boxplot(ax=axes[1], x=df['pickup_hour'], y=df['fare_amount'], showfliers=False)
    axes[1].set_xlabel('pickup hour', fontsize=9, weight='bold')
    axes[1].set_ylabel('fare amount', fontsize=9, weight='bold')
    axes[1].set_title('Graph 2 - Fare amount x pickup hour', fontsize=12)
    plt.tight_layout()
    save_figure('graph_01_02_fare_time_boxplots.png', save_dir)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, column, title in zip(
        axes,
        ['fare_amount', 'trip_distance', 'trip_duration'],
        ['Fare Amount', 'Trip Distance', 'Trip Duration'],
    ):
        sns.histplot(df[column], bins=50, kde=True, ax=ax)
        ax.set_title(f'Graph 3 - Distribution - {title}')
    plt.tight_layout()
    save_figure('graph_03_feature_distributions.png', save_dir)
    plt.close()

    df_group_sum_revenue_month = df.groupby('pickup_month')[['total_amount']].sum().round(2)
    plt.figure(figsize=(12, 5))
    data = df_group_sum_revenue_month.head(12)
    pal = sns.color_palette('Blues_d', len(data))
    rank = data['total_amount'].argsort()
    ax = sns.barplot(x=data.index, y=data['total_amount'], palette=np.array(pal[::-1])[rank])
    ax.axhline(df_group_sum_revenue_month['total_amount'].mean(), ls='-', color='red', label='global mean')
    ax.legend()
    plt.title('Graph 4 - Revenue per month', fontsize=12)
    plt.xlabel('month', fontsize=9)
    plt.ylabel('total amount', fontsize=9)
    plt.grid(True)
    save_figure('graph_04_revenue_per_month.png', save_dir)
    plt.close()


def plot_outlier_boxplots(df: pd.DataFrame, suffix: str, save_dir: str = None) -> None:
    """Plot boxplots for the main outlier-treatment columns."""
    save_dir = save_dir or config.OUTPUT_DIR
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Outlier Visualization - {suffix}', fontsize=14)

    for i, column in enumerate(config.OUTLIER_COLUMNS):
        sns.boxplot(ax=axes[i], x=df[column])
        axes[i].set_title(column)

    plt.tight_layout()
    save_figure(f'graph_05_outliers_{suffix}.png', save_dir)
    plt.close()


def plot_relationships(df: pd.DataFrame, sample_size: int = None, save_dir: str = None) -> pd.DataFrame:
    """Plot sampled scatterplots and fare-by-hour boxplots."""
    save_dir = save_dir or config.OUTPUT_DIR
    sample_size = sample_size or config.SAMPLE_SIZE
    df_sample = df.copy()
    df_sample['store_and_fwd_flag'] = df_sample['store_and_fwd_flag'].map({'N': 0, 'Y': 1})
    sample_n = min(sample_size, len(df_sample))
    df_sample = df_sample.sample(sample_n, random_state=config.RANDOM_STATE)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    sns.scatterplot(ax=axes[0], x='trip_distance', y='fare_amount', hue='VendorID', data=df_sample, alpha=0.4)
    axes[0].set_title('Graph 6 - Distance vs Fare')
    sns.scatterplot(ax=axes[1], x='trip_duration', y='fare_amount', hue='VendorID', data=df_sample, alpha=0.4)
    axes[1].set_title('Graph 7 - Duration vs Fare')
    plt.tight_layout()
    save_figure('graph_06_07_distance_duration_fare.png', save_dir)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    sns.boxplot(
        ax=axes[0],
        data=df_sample[(df_sample['pickup_hour'] >= 0) & (df_sample['pickup_hour'] <= 11)],
        x='pickup_hour',
        y='fare_amount',
        hue='store_and_fwd_flag',
    )
    axes[0].set_title('Graph 8 - Fare Amount (0h-11h)')
    sns.boxplot(
        ax=axes[1],
        data=df_sample[(df_sample['pickup_hour'] >= 12) & (df_sample['pickup_hour'] <= 23)],
        x='pickup_hour',
        y='fare_amount',
        hue='store_and_fwd_flag',
    )
    axes[1].set_title('Graph 9 - Fare Amount (12h-23h)')
    plt.tight_layout()
    save_figure('graph_08_09_fare_by_hour_flag.png', save_dir)
    plt.close()

    print(df_sample['store_and_fwd_flag'].value_counts())
    print(df_sample['VendorID'].value_counts())
    return df_sample


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str = None) -> None:
    """Plot correlation matrix for selected numeric variables."""
    save_dir = save_dir or config.OUTPUT_DIR
    columns = [
        'fare_amount',
        'trip_distance',
        'trip_duration',
        'total_amount',
        'tip_amount',
        'pickup_hour',
        'passenger_count',
        'VendorID',
    ]
    corr_df = df[columns].copy()
    corr_df['VendorID'] = pd.to_numeric(corr_df['VendorID'], errors='coerce')

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Graph 10 - Correlation Matrix')
    save_figure('graph_10_correlation_matrix.png', save_dir)
    plt.close()


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF for the independent variables used in the notebook."""
    x_vif = df[['trip_distance', 'trip_duration', 'pickup_hour', 'pickup_month']].dropna()
    x_vif_const = add_constant(x_vif)
    vif = pd.DataFrame()
    vif['feature'] = x_vif_const.columns
    vif['VIF'] = [variance_inflation_factor(x_vif_const.values, i) for i in range(x_vif_const.shape[1])]
    print(vif.to_string(index=False))
    return vif


def run_eda(df: pd.DataFrame, save_dir: str = None) -> None:
    """Runs all EDA visualizations and VIF analysis."""
    save_dir = save_dir or config.OUTPUT_DIR
    plot_initial_eda(df, save_dir)
    plot_outlier_boxplots(df, suffix='before_treatment', save_dir=save_dir)
    plot_relationships(df, save_dir=save_dir)
    plot_correlation_heatmap(df, save_dir=save_dir)
    calculate_vif(df)
    print('\nEDA Visualizations complete')
