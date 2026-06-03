import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from src import config


DAY_ORDER = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
CORRELATION_COLUMNS = [
    'fare_amount',
    'trip_distance',
    'trip_duration',
    'total_amount',
    'tip_amount',
    'pickup_hour',
    'passenger_count',
    'VendorID',
]
VIF_COLUMNS = ['trip_distance', 'trip_duration', 'pickup_hour', 'pickup_month']


def save_figure(filename: str, save_dir: str | None = None) -> None:
    """Save the current matplotlib figure when an output directory is available."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
        print(f'>> Saved: {filename}')


def calculate_iqr_limits(
    df: pd.DataFrame,
    column: str,
) -> tuple[float, float, float, float, float]:
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
    total_rows = df.shape[0]

    for column in numeric_columns:
        _, _, _, upper_limit, lower_limit = calculate_iqr_limits(df, column)
        outliers = df[(df[column] >= upper_limit) | (df[column] <= lower_limit)]
        total_outliers = outliers.shape[0]
        percentage_outliers = total_outliers / total_rows

        if 0 < percentage_outliers < 1:
            print(
                f'[ {column} ] >> {total_outliers} outliers :: '
                f'{percentage_outliers * 100:.2f}%'
            )


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


def treat_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor_limit: float = 4,
) -> pd.DataFrame:
    """Apply notebook outlier treatment to the configured numeric columns."""
    df = df.copy()
    columns = columns or config.OUTLIER_COLUMNS

    print('>> outliers count \n')
    view_outliers(df)
    print('\n>> outliers treatment \n')

    for column in columns:
        fix_data(df, column=column, factor_limit=factor_limit)

    return df


def plot_initial_eda(df: pd.DataFrame, save_dir: str | None = None) -> None:
    """Plot the notebook's first EDA charts."""
    save_dir = save_dir or config.OUTPUT_DIR

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(
        ax=axes[0],
        x=df['day_name'],
        y=df['fare_amount'],
        hue=df['VendorID'],
        order=DAY_ORDER,
        showfliers=False,
    )
    axes[0].set_xlabel('day of week', fontsize=9, weight='bold')
    axes[0].set_ylabel('fare amount', fontsize=9, weight='bold')
    axes[0].set_title('Graph 1 - Fare amount x day of week', fontsize=12)

    sns.boxplot(
        ax=axes[1],
        x=df['pickup_hour'],
        y=df['fare_amount'],
        showfliers=False,
    )
    axes[1].set_xlabel('pickup hour', fontsize=9, weight='bold')
    axes[1].set_ylabel('fare amount', fontsize=9, weight='bold')
    axes[1].set_title('Graph 2 - Fare amount x pickup hour', fontsize=12)
    plt.tight_layout()
    save_figure('graph_01_02_fare_time_boxplots.png', save_dir)
    plt.close()

    _, axes = plt.subplots(1, 3, figsize=(14, 5))
    distribution_columns = ['fare_amount', 'trip_distance', 'trip_duration']
    distribution_titles = ['Fare Amount', 'Trip Distance', 'Trip Duration']

    for ax, column, title in zip(axes, distribution_columns, distribution_titles):
        sns.histplot(df[column], bins=50, kde=True, ax=ax)
        ax.set_title(f'Graph 3 - Distribution - {title}')

    plt.tight_layout()
    save_figure('graph_03_feature_distributions.png', save_dir)
    plt.close()

    revenue_by_month = df.groupby('pickup_month')[['total_amount']].sum().round(2)
    data = revenue_by_month.head(12)
    palette = sns.color_palette('Blues_d', len(data))
    rank = data['total_amount'].argsort()

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        x=data.index,
        y=data['total_amount'],
        hue=data.index.to_list(),
        palette=np.array(palette[::-1])[rank].tolist(),
    )
    ax.axhline(
        revenue_by_month['total_amount'].mean(),
        ls='-',
        color='red',
        label='global mean',
    )
    ax.legend()
    plt.title('Graph 4 - Revenue per month', fontsize=12)
    plt.xlabel('month', fontsize=9)
    plt.ylabel('total amount', fontsize=9)
    plt.grid(True)
    save_figure('graph_04_revenue_per_month.png', save_dir)
    plt.close()


def plot_outlier_boxplots(df: pd.DataFrame, suffix: str, save_dir: str | None = None) -> None:
    """Plot boxplots for the main outlier-treatment columns."""
    save_dir = save_dir or config.OUTPUT_DIR
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Outlier Visualization - {suffix}', fontsize=14)

    for index, column in enumerate(config.OUTLIER_COLUMNS):
        sns.boxplot(ax=axes[index], x=df[column])
        axes[index].set_title(column)

    plt.tight_layout()
    save_figure(f'graph_05_outliers_{suffix}.png', save_dir)
    plt.close()


def plot_relationships(
    df: pd.DataFrame,
    sample_size: int | None = None,
    save_dir: str | None = None,
) -> pd.DataFrame:
    """Plot sampled scatterplots and fare-by-hour boxplots."""
    save_dir = save_dir or config.OUTPUT_DIR
    sample_size = sample_size or config.SAMPLE_SIZE

    df_sample = df.copy()
    df_sample['store_and_fwd_flag'] = df_sample['store_and_fwd_flag'].map({'N': 0, 'Y': 1})
    sample_n = min(sample_size, len(df_sample))
    df_sample = df_sample.sample(sample_n, random_state=config.RANDOM_STATE)

    _, axes = plt.subplots(1, 2, figsize=(18, 5))
    sns.scatterplot(
        ax=axes[0],
        data=df_sample,
        x='trip_distance',
        y='fare_amount',
        hue='VendorID',
        alpha=0.4,
    )
    axes[0].set_title('Graph 6 - Distance vs Fare')

    sns.scatterplot(
        ax=axes[1],
        data=df_sample,
        x='trip_duration',
        y='fare_amount',
        hue='VendorID',
        alpha=0.4,
    )
    axes[1].set_title('Graph 7 - Duration vs Fare')
    plt.tight_layout()
    save_figure('graph_06_07_distance_duration_fare.png', save_dir)
    plt.close()

    morning_trips = df_sample[
        (df_sample['pickup_hour'] >= 0) & (df_sample['pickup_hour'] <= 11)
    ]
    afternoon_evening_trips = df_sample[
        (df_sample['pickup_hour'] >= 12) & (df_sample['pickup_hour'] <= 23)
    ]

    _, axes = plt.subplots(1, 2, figsize=(18, 5))
    sns.boxplot(
        ax=axes[0],
        data=morning_trips,
        x='pickup_hour',
        y='fare_amount',
        hue='store_and_fwd_flag',
    )
    axes[0].set_title('Graph 8 - Fare Amount (0h-11h)')

    sns.boxplot(
        ax=axes[1],
        data=afternoon_evening_trips,
        x='pickup_hour',
        y='fare_amount',
        hue='store_and_fwd_flag',
    )
    axes[1].set_title('Graph 9 - Fare Amount (12h-23h)')
    plt.tight_layout()
    save_figure('graph_08_09_fare_by_hour_flag.png', save_dir)
    plt.close()

    return df_sample


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str | None = None) -> None:
    """Plot correlation matrix for selected numeric variables."""
    save_dir = save_dir or config.OUTPUT_DIR
    corr_df = df[CORRELATION_COLUMNS].copy()
    corr_df['VendorID'] = pd.to_numeric(corr_df['VendorID'], errors='coerce')

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Graph 10 - Correlation Matrix')
    save_figure('graph_10_correlation_matrix.png', save_dir)
    plt.close()


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF for the independent variables used in the notebook."""
    x_vif = df[VIF_COLUMNS].dropna()
    x_vif_const = add_constant(x_vif)
    vif_values = [
        variance_inflation_factor(x_vif_const.values, index)
        for index in range(x_vif_const.shape[1])
    ]

    return pd.DataFrame({'feature': x_vif_const.columns, 'VIF': vif_values})


def run_eda(df: pd.DataFrame, save_dir: str | None = None) -> None:
    """Runs all EDA visualizations and VIF analysis."""
    save_dir = save_dir or config.OUTPUT_DIR
    plot_initial_eda(df, save_dir)
    plot_outlier_boxplots(df, suffix='before_treatment', save_dir=save_dir)
    plot_relationships(df, save_dir=save_dir)
    plot_correlation_heatmap(df, save_dir=save_dir)
    calculate_vif(df)
    print('\nEDA Visualizations complete')
