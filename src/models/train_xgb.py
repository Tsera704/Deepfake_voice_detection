import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone
def _xgb_class():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError as e:
        raise ImportError("xgboost not installed. Run: pip install -r requirements.txt (Colab/Kaggle cell 1)") from e


def _select_k(cfg, n_feat):
    k = cfg.get("feature_select", {}).get("mutual_info_k", 200)
    return max(1, min(int(k), int(n_feat)))


def build_pipeline(cfg, n_feat, params=None):
    vt = cfg.get("feature_select", {}).get("variance_threshold", 1e-4)
    k = _select_k(cfg, n_feat)
    rs = cfg.get("training", {}).get("random_state", 42)
    clf_params = dict(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", tree_method="hist",
        random_state=rs, n_jobs=-1,
    )
    if params:
        clf_params.update(params)
    XGBClassifier = _xgb_class()
    return Pipeline([
        ("scaler", StandardScaler()),
        ("var", VarianceThreshold(threshold=vt)),
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("clf", XGBClassifier(**clf_params)),
    ])


def _suggest(trial, space):
    lo, hi = space.get("n_estimators", [100, 500])
    md_lo, md_hi = space.get("max_depth", [4, 12])
    lr_lo, lr_hi = space.get("learning_rate", [0.01, 0.1])
    ss_lo, ss_hi = space.get("subsample", [0.6, 1.0])
    return {
        "n_estimators": trial.suggest_int("n_estimators", int(lo), int(hi)),
        "max_depth": trial.suggest_int("max_depth", int(md_lo), int(md_hi)),
        "learning_rate": trial.suggest_float("learning_rate", float(lr_lo), float(lr_hi), log=True),
        "subsample": trial.suggest_float("subsample", float(ss_lo), float(ss_hi)),
    }


def tune_and_train(X, y, feat_cfg, model_cfg, n_trials=5, output_dir="models", experiment="smoke"):
    import optuna
    from src.evaluation.metrics import summarize
    os.makedirs(output_dir, exist_ok=True)
    X = np.asarray(X)
    y = np.asarray(y)
    rs = feat_cfg.get("training", {}).get("random_state", 42)
    cv_folds = model_cfg.get("xgboost", {}).get("cv_folds", 5)
    space = model_cfg.get("xgboost", {}).get("optuna_study", {}).get("param_space", {})
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        pipe = build_pipeline(feat_cfg, X.shape[1], _suggest(trial, space))
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        return summarize(y, proba)["eer"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best = build_pipeline(feat_cfg, X.shape[1], study.best_params)
    best.fit(X, y)

    try:
        import mlflow
        uri = model_cfg.get("mlflow", {}).get("tracking_uri", os.path.join(output_dir, "mlruns"))
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(model_cfg.get("mlflow", {}).get("experiment_name", "explainable_deepfake_detection"))
        with mlflow.start_run(run_name=f"xgb_{experiment}"):
            mlflow.log_params(study.best_params)
            proba = cross_val_predict(best, X, y, cv=cv, method="predict_proba")[:, 1]
            mlflow.log_metrics({k: float(v) for k, v in summarize(y, proba).items() if np.isfinite(v)})
    except Exception as e:
        print(f"[MLflow skipped] {e}")

    import pickle
    out_path = os.path.join(output_dir, f"xgb_{experiment}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"pipeline": best, "params": study.best_params, "n_feat": X.shape[1]}, f)
    proba = cross_val_predict(clone(best), X, y, cv=cv, method="predict_proba")[:, 1]
    return best, summarize(y, proba), out_path
