import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, learning_curve
from src import config


def regression_metrics(y_true, y_pred) -> dict:
    """Return the core regression metrics used in the notebook."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_true, y_pred),
    }


def evaluate_models(best_models: dict, x_train_scaled, x_test_scaled, y_train, y_test) -> tuple:
    """Evaluate trained regressors and select the best model by test R2."""
    best_score = -np.inf
    best_model_name = None

    for name, model in best_models.items():
        train_metrics = regression_metrics(y_train, model.predict(x_train_scaled))
        test_metrics = regression_metrics(y_test, model.predict(x_test_scaled))

        print(f">> {name} - Training metrics:")
        print(f"   R2:   {train_metrics['R2']:.4f}")
        print(f"   RMSE: {train_metrics['RMSE']:.4f}")
        print(f"   MAE:  {train_metrics['MAE']:.4f}")

        print(f">> {name} - Test metrics:")
        print(f"   R2:   {test_metrics['R2']:.4f}")
        print(f"   RMSE: {test_metrics['RMSE']:.4f}")
        print(f"   MAE:  {test_metrics['MAE']:.4f}")
        print(f"[!] R2 difference (test - train): {(test_metrics['R2'] - train_metrics['R2']):.4f}\n")

        if test_metrics['R2'] > best_score:
            best_score = test_metrics['R2']
            best_model_name = name

    print(f">> Best model: {best_model_name} with R2 = {best_score:.4f}\n")
    return best_model_name, best_models[best_model_name]


def plot_model_diagnostics(best_model, best_model_name: str, x_train_scaled, x_test_scaled, y_train, y_test, save_dir: str = None) -> None:
    """Plot true-vs-predicted and learning-curve diagnostics."""
    save_dir = save_dir or config.OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    cv = KFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    y_best_pred = best_model.predict(x_test_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].scatter(y_test, y_best_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'Graph 11 - {best_model_name}: True vs Predicted')
    axes[0].grid(True)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=best_model,
        X=x_train_scaled,
        y=y_train,
        cv=cv,
        scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )
    axes[1].plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training R2')
    axes[1].plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation R2')
    axes[1].set_xlabel('Training Set Size')
    axes[1].set_ylabel('R2 Score')
    axes[1].set_title(f'Graph 12 - Learning Curve - {best_model_name}')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    filename = 'graph_11_12_model_diagnostics.png'
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
    print(f'>> Saved: {filename}')
    plt.close()


def plot_residual_diagnostics(model, x_test_scaled, y_test, save_dir: str = None) -> pd.DataFrame:
    """Plot actual-vs-predicted, residual distribution, and residual scatter."""
    save_dir = save_dir or config.OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    y_pred_test = model.predict(x_test_scaled)
    results = pd.DataFrame({'actual': y_test.values, 'predicted': y_pred_test.ravel()})
    results['residual'] = results['actual'] - results['predicted']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.set(style='whitegrid')
    sns.scatterplot(x='actual', y='predicted', data=results, s=20, alpha=0.5, ax=axes[0])
    axes[0].set_title('Graph 13 - Actual vs Predicted')
    sns.histplot(results['residual'], bins=np.arange(-15, 15.5, 0.5), ax=axes[1])
    axes[1].set_title('Graph 14 - Distribution of Residuals')
    sns.scatterplot(x='predicted', y='residual', data=results, ax=axes[2])
    axes[2].axhline(0, c='red')
    axes[2].set_title('Graph 15 - Residuals over Predicted Values')
    plt.tight_layout()
    filename = 'graph_13_15_residual_diagnostics.png'
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
    print(f'>> Saved: {filename}')
    plt.close()
    return results


def segment_metrics(x_test: pd.DataFrame, y_test: pd.Series, y_pred, bins=None, labels=None) -> dict:
    """Calculate R2, RMSE, and MAE by trip-distance segment."""
    bins = bins or [0, 2, 10, 1e9]
    labels = labels or ['short', 'medium', 'long']
    seg = pd.cut(x_test['trip_distance'], bins=bins, labels=labels)
    res = {}
    y_pred_series = pd.Series(y_pred, index=y_test.index)

    for label in labels:
        idx = seg[seg == label].index
        if len(idx) < 3:
            continue
        yt = y_test.loc[idx]
        yp = y_pred_series.loc[idx]
        res[label] = {
            'N': len(idx),
            'R2': round(r2_score(yt, yp), 4),
            'RMSE': round(np.sqrt(mean_squared_error(yt, yp)), 4),
            'MAE': round(mean_absolute_error(yt, yp), 4),
        }
    return res


def run_fast_model_evaluation(x_train, x_test, y_train, y_test) -> None:
    """Run the notebook's quick HistGradientBoostingRegressor segment evaluation."""
    model = HistGradientBoostingRegressor(random_state=config.RANDOM_STATE)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    metrics = regression_metrics(y_test, y_pred)

    print(f"R2:   {metrics['R2']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print('\nSegment metrics (trip_distance bins):')
    pprint(segment_metrics(x_test, y_test, y_pred), sort_dicts=False)

    hour_stats = {}
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    for h in sorted(x_test['pickup_hour'].dropna().unique()):
        idx = x_test[x_test['pickup_hour'] == h].index
        if len(idx) < 5:
            continue
        hour_stats[int(h)] = {'n': len(idx), 'R2': round(r2_score(y_test.loc[idx], y_pred_series.loc[idx]), 4)}

    print('\nExemplo métricas por hora (rmse):')
    pprint(list(hour_stats.items())[6:21], sort_dicts=False)
