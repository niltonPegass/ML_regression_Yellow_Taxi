import os
import pickle
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from src import config


def prepare_modeling_data(df: pd.DataFrame):
    """Prepare final model features, train/test split, and scaled matrices."""
    df_model = df.drop(columns=config.DROP_COLUMNS, axis=1, errors='ignore')
    df_model = df_model.dropna(subset=[config.TARGET_COLUMN])
    df_model = pd.get_dummies(df_model, drop_first=True)

    features = df_model.drop(config.TARGET_COLUMN, axis=1)
    target = df_model[config.TARGET_COLUMN]
    print(f'feature columns:\n{features.columns.to_list()}')

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, x_train, x_test, scaler


def get_models_and_grids() -> tuple:
    """Initialize candidate regressors and notebook hyperparameter grids."""
    models = {
        'Linear Regression': LinearRegression(),
        'HistGBR': HistGradientBoostingRegressor(random_state=config.RANDOM_STATE),
        'Decision Tree': DecisionTreeRegressor(random_state=config.RANDOM_STATE),
        'Random Forest': RandomForestRegressor(random_state=config.RANDOM_STATE),
        'XGBoost': xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=config.RANDOM_STATE,
        ),
    }

    param_grids = {
        'Decision Tree': {
            'max_depth': [1, 4, None],
            'min_samples_leaf': [2, 5, 10],
            'min_samples_split': [2, 10],
        },
        'Random Forest': {
            'max_depth': [1, 4, None],
            'min_samples_leaf': [2, 5, 10],
            'min_samples_split': [2, 10],
            'max_features': [0.5, 1.0],
            'max_samples': [0.7, 1.0],
            'n_estimators': [50, 200],
        },
        'XGBoost': {
            'max_depth': [1, 4, 6],
            'subsample': [0.5, 1.0],
            'min_child_weight': [2, 5],
            'learning_rate': [0.1, 0.2],
            'n_estimators': [50, 200],
        },
    }
    return models, param_grids


def train_and_optimize_models(x_train_scaled, y_train) -> dict:
    """Train direct models and tune tree/boosting regressors with GridSearchCV."""
    print('Starting model training and hyperparameter optimization [...]\n')
    models, param_grids = get_models_and_grids()
    cv = KFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    best_models = {}

    for name, model in models.items():
        if name in param_grids:
            print(f'>> Tuning and training {name}')
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                scoring='r2',
                cv=cv,
                n_jobs=-1,
                verbose=0,
            )
            grid.fit(x_train_scaled, y_train)
            best_models[name] = grid.best_estimator_
            print(f'└> {name} best parameters:')
            for param, value in grid.best_params_.items():
                print(f'    {param}: {value}')
        else:
            print(f'>> Training {name}')
            model.fit(x_train_scaled, y_train)
            best_models[name] = model

    print('\nModel training complete')
    return best_models


def save_artifacts(best_models: dict, scaler: StandardScaler, save_dir: str = None) -> None:
    """Save trained models and scaler as pickle files."""
    save_dir = save_dir or config.MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)

    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'>> Saved scaler: {scaler_path}')

    for name, model in best_models.items():
        model_name_clean = name.lower().replace(' ', '_')
        model_path = os.path.join(save_dir, f'model_{model_name_clean}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    print(f'>> Saved models: {save_dir}')
