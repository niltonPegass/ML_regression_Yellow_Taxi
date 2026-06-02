import os


# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default Kaggle dataset path used in the original notebook.
DEFAULT_KAGGLE_DATA_PATH = '/kaggle/input/new-york-city-taxi-trips-2017/2017_Yellow_Taxi_Trip_Data.csv'

# Local dataset path for running the project outside Kaggle.
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', '2017_Yellow_Taxi_Trip_Data.csv')

# Path to directory where output figures will be saved.
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')

# Path to directory where trained models will be saved.
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model configurations
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
TARGET_COLUMN = 'fare_amount'
SAMPLE_SIZE = 20000

# Columns used for outlier treatment in the notebook.
OUTLIER_COLUMNS = ['trip_distance', 'fare_amount', 'trip_duration']

# Columns removed before final model training.
DROP_COLUMNS = [
    'pickup_dropoff',
    'mean_trip_duration',
    'mean_trip_distance',
    'is_weekend',
    'Unnamed: 0',
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime',
    'store_and_fwd_flag',
    'PULocationID',
    'DOLocationID',
    'payment_type',
    'extra',
    'mta_tax',
    'tip_amount',
    'tolls_amount',
    'improvement_surcharge',
    'total_amount',
]
