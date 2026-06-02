import pandas as pd


def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create datetime, trip-duration, and base temporal features."""
    df = df.copy()
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    df['trip_duration'] = (
        (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'])
        .dt.total_seconds()
        .div(60)
        .round(2)
    )
    df['trip_distance'] = pd.to_numeric(df['trip_distance'], errors='coerce')
    df['day_name'] = df['tpep_pickup_datetime'].dt.day_name().str[:3].str.lower()
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
    df['is_weekend'] = df['day_name'].isin(['sat', 'sun'])

    for col in ['RatecodeID', 'VendorID']:
        df[col] = df[col].astype('str')

    return df


def rush_transformation(row: pd.Series) -> int:
    """Return 1 when a trip happened during weekday rush-hour windows."""
    condition_weekday = row['day_name'] not in ['sat', 'sun']
    cond_06_09 = 6 <= row['tpep_pickup_datetime'].hour <= 9
    cond_17_20 = 17 <= row['tpep_pickup_datetime'].hour <= 20
    return int(condition_weekday and (cond_06_09 or cond_17_20))


def day_period(row: pd.Series) -> str:
    """Classify pickup time into administrative, dawn, or night periods."""
    pickup_hour = row['tpep_pickup_datetime'].hour
    if 6 <= pickup_hour < 18:
        return 'adm_period'
    if 0 <= pickup_hour < 6:
        return 'dawntime'
    return 'nighttime'


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rush-hour, day-period, route, and route-average features."""
    df = df.copy()
    df['rush_hour'] = df.apply(rush_transformation, axis=1)
    df['day_period'] = df.apply(day_period, axis=1)
    df['pickup_dropoff'] = df['PULocationID'].astype('str') + ' ' + df['DOLocationID'].astype('str')

    for column in ['trip_distance', 'trip_duration']:
        grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[[column]].round(4)
        df[f'mean_{column}'] = df['pickup_dropoff'].map(grouped[column].to_dict())

    return df
