from src import config
from src.data_loader import display_dataset_overview, load_data
from src.eda import plot_outlier_boxplots, run_eda, treat_outliers
from src.feature_engineering import create_advanced_features, create_base_features
from src.insights import analyze_payment_type_fares
from src.model_evaluation import (
    evaluate_models,
    plot_model_diagnostics,
    plot_residual_diagnostics,
    run_fast_model_evaluation,
)
from src.model_training import prepare_modeling_data, save_artifacts, train_and_optimize_models


def main():
    print('=' * 80)
    print('STARTING NYC YELLOW TAXI FARE REGRESSION PIPELINE')
    print('=' * 80)

    print('\n' + '=' * 50)
    print('STEP 1: DATA INGESTION AND DATASET OVERVIEW')
    print('=' * 50)
    df = load_data()
    display_dataset_overview(df)

    print('\n' + '=' * 50)
    print('STEP 2: BASE FEATURE ENGINEERING')
    print('=' * 50)
    df = create_base_features(df)
    print('Base features created successfully')

    print('\n' + '=' * 50)
    print('STEP 3: EXPLORATORY DATA ANALYSIS')
    print('=' * 50)
    run_eda(df, save_dir=config.OUTPUT_DIR)

    print('\n' + '=' * 50)
    print('STEP 4: OUTLIER TREATMENT')
    print('=' * 50)
    df = treat_outliers(df)
    plot_outlier_boxplots(df, suffix='after_treatment', save_dir=config.OUTPUT_DIR)

    print('\n' + '=' * 50)
    print('STEP 5: ADVANCED FEATURE ENGINEERING')
    print('=' * 50)
    df = create_advanced_features(df)
    print('Advanced features created successfully')

    print('\n' + '=' * 50)
    print('STEP 6: SPLITTING AND SCALING FEATURES')
    print('=' * 50)
    x_train_scaled, x_test_scaled, y_train, y_test, x_train, x_test, scaler = prepare_modeling_data(df)
    print(f'Features scaled successfully')
    print(f'>> Train shape: {x_train_scaled.shape}')
    print(f'>> Test shape: {x_test_scaled.shape}')

    print('\n' + '=' * 50)
    print('STEP 7: FAST MODEL EVALUATION')
    print('=' * 50)
    run_fast_model_evaluation(x_train, x_test, y_train, y_test)

    print('\n' + '=' * 50)
    print('STEP 8: TRAINING AND HYPERPARAMETER OPTIMIZATION')
    print('=' * 50)
    best_models = train_and_optimize_models(x_train_scaled, y_train)

    print('\n' + '=' * 50)
    print('STEP 9: SAVING TRAINED MODEL ARTIFACTS')
    print('=' * 50)
    save_artifacts(best_models, scaler)

    print('\n' + '=' * 50)
    print('STEP 10: EVALUATING MODELS')
    print('=' * 50)
    best_model_name, best_model = evaluate_models(best_models, x_train_scaled, x_test_scaled, y_train, y_test)

    print('\n' + '=' * 50)
    print('STEP 11: GENERATING EVALUATION CHARTS')
    print('=' * 50)
    plot_model_diagnostics(best_model, best_model_name, x_train_scaled, x_test_scaled, y_train, y_test)
    plot_residual_diagnostics(best_model, x_test_scaled, y_test)

    print('\n' + '=' * 50)
    print('STEP 12: PAYMENT TYPE INSIGHTS')
    print('=' * 50)
    analyze_payment_type_fares(df)

    print('\n' + '=' * 80)
    print('PIPELINE COMPLETED SUCCESSFULLY!')
    print(f'>> All figures have been saved:\n{config.OUTPUT_DIR}')
    print(f'>> Model artifacts have been saved:\n{config.MODELS_DIR}')
    print('=' * 80)


if __name__ == '__main__':
    main()
