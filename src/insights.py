import pandas as pd
from scipy import stats


def analyze_payment_type_fares(df: pd.DataFrame) -> tuple:
    """Compare fare amount for credit-card and cash payments using a Welch t-test."""
    df_credit = df[df['payment_type'] == 1]
    df_cash = df[df['payment_type'] == 2]

    mean_credit = df_credit['fare_amount'].mean()
    mean_cash = df_cash['fare_amount'].mean()
    std_credit = df_credit['fare_amount'].std()
    std_cash = df_cash['fare_amount'].std()

    print(f'mean fare amount [credit]: {mean_credit:.2f} // mean fare amount [cash]: {mean_cash:.2f}')
    print(f'std fare amount [credit]: {std_credit:.2f} // std fare amount [cash]: {std_cash:.2f}')

    statistic, pvalue = stats.ttest_ind(
        a=df_credit['fare_amount'],
        b=df_cash['fare_amount'],
        equal_var=False,
    )
    print(f'statistic: {statistic:.2f} // pvalue: {pvalue * 100:.2f}%')
    return statistic, pvalue
