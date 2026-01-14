import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import logging

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from typing import Optional

import warnings

warnings.simplefilter("ignore", ConvergenceWarning)

def regression_metrics(y_true, y_pred, mape_eps=1e-8):
    """
    Returns: R2, MAE, RMSE, MAPE
    MAPE uses an epsilon to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denom = np.maximum(np.abs(y_true), mape_eps)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}

def build_xy(
    df: pd.DataFrame,
    target: str,
    interaction_feature:Optional[str],
    date_col: str = "Date",
    pollutant_cols=("Ozone", "NO2", "PM2.5", "CO"),
    interaction_prefix: str = "int_",
    drop_other_pollutants: bool = True,
):
    """
    interaction_feature:
      - None  -> baseline model (drop ALL interaction terms)
      - str   -> keep ONLY this interaction term, drop all others
    """
    data = df.copy()

    if date_col not in data.columns:
        raise ValueError(f"'{date_col}' column not found.")
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)

    if target not in data.columns:
        raise ValueError(f"Target '{target}' not found.")

    # -----------------------------
    # Handle interaction features
    # -----------------------------
    interaction_cols = [c for c in data.columns if c.startswith(interaction_prefix)]

    if interaction_feature is None:
        # BASELINE: drop ALL interaction columns
        data = data.drop(columns=interaction_cols, errors="ignore")
    else:
        # SINGLE interaction: keep only the requested one
        if interaction_feature not in data.columns:
            raise ValueError(f"Interaction feature '{interaction_feature}' not found.")
        drop_cols = [c for c in interaction_cols if c != interaction_feature]
        data = data.drop(columns=drop_cols, errors="ignore")

    # -----------------------------
    # Build y
    # -----------------------------
    y = data[target].astype(float)

    # -----------------------------
    # Build X
    # -----------------------------
    drop_cols = [date_col, target]

    pollutant_cols = [c for c in pollutant_cols if c in data.columns]
    if drop_other_pollutants:
        drop_cols += [c for c in pollutant_cols if c != target]

    X = data.drop(columns=drop_cols, errors="ignore")

    # ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # drop rows where y is missing
    out = pd.DataFrame({date_col: data[date_col], "y": y})
    out = pd.concat([out, X], axis=1).dropna(subset=["y"]).reset_index(drop=True)

    return out


def fill_exog_missing(X: pd.DataFrame) -> pd.DataFrame:
    """
    Common time-series friendly filling:
    - forward fill
    - backward fill
    - remaining NaN -> 0
    """
    return X.ffill().bfill().fillna(0.0)


def align_train_test_exog(train_xy: pd.DataFrame, test_xy: pd.DataFrame):
    """
    SARIMAX requires train and forecast exog to have SAME columns in SAME order.
    This function:
    - unions columns
    - reindexes both to the same set
    - fills missing values
    """
    X_tr = train_xy.drop(columns=["Date", "y"], errors="ignore")
    X_te = test_xy.drop(columns=["Date", "y"], errors="ignore")

    all_cols = sorted(set(X_tr.columns).union(set(X_te.columns)))

    X_tr = X_tr.reindex(columns=all_cols, fill_value=0.0)
    X_te = X_te.reindex(columns=all_cols, fill_value=0.0)

    X_tr = fill_exog_missing(X_tr)
    X_te = fill_exog_missing(X_te)

    y_tr = train_xy["y"].to_numpy(dtype=float)
    y_te = test_xy["y"].to_numpy(dtype=float)

    return y_tr, X_tr.to_numpy(dtype=float), y_te, X_te.to_numpy(dtype=float), all_cols


def fit_and_forecast_sarimax(y_train, X_train, X_forecast, order, trend="c"):
    """
    Fits SARIMAX(y_train ~ X_train) then forecasts len(X_forecast) steps using X_forecast.

    order = (p, d, q):
      p = AR order (how many lagged y terms)
      d = differencing order (how many times to difference y to make it stationary)
      q = MA order (how many lagged forecast errors)

    trend='c' adds an intercept term.
    """
    model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=len(X_forecast), exog=X_forecast).predicted_mean
    return np.asarray(pred, dtype=float), res


def select_order_by_cv_rmse(y_tr, X_tr, p_values, d_values, q_values, n_splits=5, trend="c"):
    """
    Expanding-window time-series CV using TimeSeriesSplit:
    Fold 1: train [0..t1] validate [t1+1..t2]
    Fold 2: train [0..t2] validate [t2+1..t3]
    ...
    This is a very common forecasting evaluation scheme (rolling-origin / expanding window).

    Selects the order (p,d,q) with the lowest MEAN validation RMSE.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    orders = [(p, d, q) for p in p_values for d in d_values for q in q_values]
    order_to_rmses = {order: [] for order in orders}

    for tr_idx, va_idx in tscv.split(X_tr):
        y_train, y_val = y_tr[tr_idx], y_tr[va_idx]
        X_train, X_val = X_tr[tr_idx], X_tr[va_idx]

        for order in orders:
            try:
                y_hat, _ = fit_and_forecast_sarimax(y_train, X_train, X_val, order=order, trend=trend)
                rmse = np.sqrt(mean_squared_error(y_val, y_hat))
                order_to_rmses[order].append(rmse)
            except Exception:
                order_to_rmses[order].append(np.inf)

    mean_rmse = {order: float(np.mean(v)) for order, v in order_to_rmses.items()}
    best_order = min(mean_rmse, key=mean_rmse.get)
    return best_order, mean_rmse

def cv_metrics_for_order(y_tr, X_tr, order, n_splits=5, trend="c"):
    """
    After selecting 'order', compute CV metrics (R2, MAE, RMSE, MAPE) across folds.
    Returns mean and std over folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for tr_idx, va_idx in tscv.split(X_tr):
        y_train, y_val = y_tr[tr_idx], y_tr[va_idx]
        X_train, X_val = X_tr[tr_idx], X_tr[va_idx]

        y_hat, _ = fit_and_forecast_sarimax(y_train, X_train, X_val, order=order, trend=trend)
        fold_metrics.append(regression_metrics(y_val, y_hat))

    keys = fold_metrics[0].keys()
    mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std = {k: float(np.std([m[k] for m in fold_metrics], ddof=1)) for k in keys}

    return mean, std, fold_metrics

def fit_on_train_and_test(y_tr, X_tr, y_te, X_te, order, trend="c"):
    """
    Final step:
    - Fit on ALL training data
    - Forecast for the FULL test horizon
    - Compute test metrics once (no leakage, true holdout)
    """
    y_hat, fitted_res = fit_and_forecast_sarimax(y_tr, X_tr, X_te, order=order, trend=trend)
    test_metrics = regression_metrics(y_te, y_hat)
    return test_metrics, y_hat, fitted_res


def run_armax_from_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    interaction_feature: str,
    n_splits: int = 5,
    p_max: int = 3,
    q_max: int = 3,
    trend: str = "c",
):
    """
    ARMAX means ARMA + exogenous:
      order = (p, 0, q)
    We grid-search p in [0..p_max], q in [0..q_max], d fixed at 0.
    """
    train_xy = build_xy(train_df, target=target, interaction_feature=interaction_feature)
    test_xy  = build_xy(test_df,  target=target, interaction_feature=interaction_feature)

    y_tr, X_tr, y_te, X_te, exog_cols = align_train_test_exog(train_xy, test_xy)

    p_values = range(0, p_max + 1)
    d_values = [0]
    q_values = range(0, q_max + 1)

    best_order, mean_rmse_table = select_order_by_cv_rmse(
        y_tr, X_tr, p_values, d_values, q_values, n_splits=n_splits, trend=trend
    )

    cv_mean, cv_std, _ = cv_metrics_for_order(y_tr, X_tr, order=best_order, n_splits=n_splits, trend=trend)

    test_metrics, test_pred, fitted_model = fit_on_train_and_test(
        y_tr, X_tr, y_te, X_te, order=best_order, trend=trend
    )

    return {
        "model_type": "ARMAX (SARIMAX with d=0)",
        "target": target,
        "interaction_feature": interaction_feature,
        "best_order_(p,d,q)": best_order,
        "cv_metrics_mean": cv_mean,
        "cv_metrics_std": cv_std,
        "test_metrics": test_metrics,
        "test_pred": test_pred,
        "exog_columns_used": exog_cols,
        "order_mean_cv_rmse": mean_rmse_table,
        "fitted_model": fitted_model,
    }


def run_arimax_from_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    interaction_feature: str,
    n_splits: int = 5,
    p_max: int = 3,
    d_max: int = 1,
    q_max: int = 3,
    trend: str = "c",
):
    """
    ARIMAX means ARIMA + exogenous:
      order = (p, d, q)
    We grid-search:
      p in [0..p_max], d in [0..d_max], q in [0..q_max].
    """
    train_xy = build_xy(train_df, target=target, interaction_feature=interaction_feature)
    test_xy  = build_xy(test_df,  target=target, interaction_feature=interaction_feature)

    y_tr, X_tr, y_te, X_te, exog_cols = align_train_test_exog(train_xy, test_xy)

    p_values = range(0, p_max + 1)
    d_values = range(0, d_max + 1)
    q_values = range(0, q_max + 1)

    best_order, mean_rmse_table = select_order_by_cv_rmse(
        y_tr, X_tr, p_values, d_values, q_values, n_splits=n_splits, trend=trend
    )

    cv_mean, cv_std, _ = cv_metrics_for_order(y_tr, X_tr, order=best_order, n_splits=n_splits, trend=trend)

    test_metrics, test_pred, fitted_model = fit_on_train_and_test(
        y_tr, X_tr, y_te, X_te, order=best_order, trend=trend
    )

    return {
        "model_type": "ARIMAX (SARIMAX with d>=0)",
        "target": target,
        "interaction_feature": interaction_feature,
        "best_order_(p,d,q)": best_order,
        "cv_metrics_mean": cv_mean,
        "cv_metrics_std": cv_std,
        "test_metrics": test_metrics,
        "test_pred": test_pred,
        "exog_columns_used": exog_cols,
        "order_mean_cv_rmse": mean_rmse_table,
        "fitted_model": fitted_model,
    }


def main():
    log_path = "armax_arimax_run.log"

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    logging.info("Starting ARMAX / ARIMAX evaluation")

    train_df = pd.read_csv('pollution_train_2012_2020_scaled_interactions.csv')
    test_df  = pd.read_csv('pollution_test_2021_2023_scaled_interactions.csv')

    pollutants_ls = ['Ozone', 'NO2', 'PM2.5', 'CO']
    interaction_features_ls = [
        None,
        'int_traffic_humidity',
        'int_traffic_precip',
        'int_traffic_wind',
        'int_traffic_temp'
    ]

    results_rows = []

    for inter_feat in interaction_features_ls:
        interaction_name = "baseline" if inter_feat is None else inter_feat

        logging.info("#" * 80)
        logging.info(f"Interaction feature -> {interaction_name}")

        for pollutant in pollutants_ls:
            logging.info("*" * 80)
            logging.info(f"Pollutant -> {pollutant}")

            # ======================
            # ARMAX
            # ======================
            try:
                armax_res = run_armax_from_splits(
                    train_df=train_df,
                    test_df=test_df,
                    target=pollutant,
                    interaction_feature=inter_feat,
                    n_splits=5,
                    p_max=3,
                    q_max=3
                )

                p, d, q = armax_res["best_order_(p,d,q)"]

                logging.info(
                    f"[ARMAX] {pollutant} | {interaction_name} | "
                    f"order=({p},{d},{q}) | "
                    f"CV_R2={armax_res['cv_metrics_mean']['R2']:.4f} | "
                    f"TEST_R2={armax_res['test_metrics']['R2']:.4f}"
                )

                results_rows.append({
                    "model": "ARMAX",
                    "pollutant": pollutant,
                    "interaction": interaction_name,
                    "order_p": p,
                    "order_d": d,
                    "order_q": q,

                    "cv_r2_mean": armax_res["cv_metrics_mean"]["R2"],
                    "cv_mae_mean": armax_res["cv_metrics_mean"]["MAE"],
                    "cv_rmse_mean": armax_res["cv_metrics_mean"]["RMSE"],
                    "cv_mape_mean": armax_res["cv_metrics_mean"]["MAPE"],

                    "cv_r2_std": armax_res["cv_metrics_std"]["R2"],
                    "cv_mae_std": armax_res["cv_metrics_std"]["MAE"],
                    "cv_rmse_std": armax_res["cv_metrics_std"]["RMSE"],
                    "cv_mape_std": armax_res["cv_metrics_std"]["MAPE"],

                    "test_r2": armax_res["test_metrics"]["R2"],
                    "test_mae": armax_res["test_metrics"]["MAE"],
                    "test_rmse": armax_res["test_metrics"]["RMSE"],
                    "test_mape": armax_res["test_metrics"]["MAPE"],
                })

            except Exception as e:
                logging.info(f"[ARMAX FAILED] {pollutant} | {interaction_name} | {e}")

            # ======================
            # ARIMAX
            # ======================
            try:
                arimax_res = run_arimax_from_splits(
                    train_df=train_df,
                    test_df=test_df,
                    target=pollutant,
                    interaction_feature=inter_feat,
                    n_splits=5,
                    p_max=3,
                    d_max=1,
                    q_max=3
                )

                p, d, q = arimax_res["best_order_(p,d,q)"]

                logging.info(
                    f"[ARIMAX] {pollutant} | {interaction_name} | "
                    f"order=({p},{d},{q}) | "
                    f"CV_R2={arimax_res['cv_metrics_mean']['R2']:.4f} | "
                    f"TEST_R2={arimax_res['test_metrics']['R2']:.4f}"
                )

                results_rows.append({
                    "model": "ARIMAX",
                    "pollutant": pollutant,
                    "interaction": interaction_name,
                    "order_p": p,
                    "order_d": d,
                    "order_q": q,

                    "cv_r2_mean": arimax_res["cv_metrics_mean"]["R2"],
                    "cv_mae_mean": arimax_res["cv_metrics_mean"]["MAE"],
                    "cv_rmse_mean": arimax_res["cv_metrics_mean"]["RMSE"],
                    "cv_mape_mean": arimax_res["cv_metrics_mean"]["MAPE"],

                    "cv_r2_std": arimax_res["cv_metrics_std"]["R2"],
                    "cv_mae_std": arimax_res["cv_metrics_std"]["MAE"],
                    "cv_rmse_std": arimax_res["cv_metrics_std"]["RMSE"],
                    "cv_mape_std": arimax_res["cv_metrics_std"]["MAPE"],

                    "test_r2": arimax_res["test_metrics"]["R2"],
                    "test_mae": arimax_res["test_metrics"]["MAE"],
                    "test_rmse": arimax_res["test_metrics"]["RMSE"],
                    "test_mape": arimax_res["test_metrics"]["MAPE"],
                })

            except Exception as e:
                logging.info(f"[ARIMAX FAILED] {pollutant} | {interaction_name} | {e}")

    # ======================
    # Save CSV
    # ======================
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv("armax_arimax_all_results.csv", index=False)

    logging.info("Finished all experiments")
    logging.info("Saved results to armax_arimax_all_results.csv")

if __name__ == "__main__":
    main()