import logging
import os
from argparse import ArgumentParser, Namespace
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (INFO, DEBUG, etc.)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Customize the format
)

download_path = "./allstate_claims_severity"
cv_rs = 42 # default random state for cross-validation
cb_rs = 42 # default random state for CatBoostRegressor
shuffle_rs = 42 # default random state for shuffling data

def download_kaggle_data(download_path: str) -> None:
    api = KaggleApi()
    api.authenticate()
    competition_name = "allstate-claims-severity"
    os.makedirs(download_path, exist_ok=True)
    api.competition_download_files(competition_name, path=download_path)
    import zipfile
    with zipfile.ZipFile(f"{download_path}/{competition_name}.zip", "r") as zip_ref:
        zip_ref.extractall(download_path)


def load_data(download_path: str, sample_size:int|None) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    df = pd.read_csv(f"{download_path}/train.csv")
    assert (sample_size < 188_000) or (sample_size is None), "Sample size should be less than 188K"
    if sample_size is not None:
        df = df.sample(sample_size)
    cat_features = [c for c in df.columns if c.startswith('cat')]
    cont_features = [c for c in df.columns if c.startswith('cont')]
    feature_names = cat_features + cont_features
    return df, cat_features, cont_features, feature_names


def get_g_rmse(df: pd.DataFrame, k: int, cv_rs: int, cb_rs: int, shuffle_rs: int) -> float:
    """"
    Calculate RMSE for g-th run of cross-validation
    :param df: pandas DataFrame with data
    :param k: number of cross-validation
    :param cv_rs: random state for cross-validation
    :param cb_rs: random state for CatBoostRegressor
    :param shuffle_rs: random state for shuffling data
    :return: RMSE for g-th run of cross-validation
    """
    df = df.sample(frac=1, random_state=shuffle_rs) # shuffle data for more stable GBM estimation
    X = df[feature_names]
    y = df['loss']
    kf = KFold(n_splits=k, shuffle=True, random_state=cv_rs)
    oof_predictions = np.zeros(len(df))
    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X, y))):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
        model = CatBoostRegressor(
            task_type="CPU",
            loss_function='RMSE',
            use_best_model=True,
            eval_metric='RMSE',
            random_seed=cb_rs,
            iterations=1000,
            eval_fraction=0.2,
            logging_level='Silent',
            thread_count=20,
        )
        model.fit(train_pool)
        fold_predictions = model.predict(X_val)
        oof_predictions[val_idx] = fold_predictions
    rmse_g_i = root_mean_squared_error(y, oof_predictions)
    return rmse_g_i


def cli_args()-> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--download", help="download train data", default=False, action='store_true')
    parser.add_argument("--randomize-cb-rs", help="Stable random seed for Random forest estimation."
                                               " If False, then randomness of RF estimation is added", default=False,
                        action='store_true')
    parser.add_argument("--randomize-cv-rs", help="Stable random seed for cross-validation splits"
                                               " If False, then randomness of cross-validation estimation is added",
                        default=False,
                        action='store_true')
    parser.add_argument("--randomize-shuffle-rs", help="Stable random seed for cross-validation splits"
                                               " If False, then randomness of cross-validation estimation is added",
                        default=False,
                        action='store_true')
    parser.add_argument("--g-count", help="Maximum number cv-run", type=int, default=3)
    parser.add_argument("--sample-size", help="Random sampling for small-sample experiments, use a number less than 188K",
                        type=int, default=None)
    parser.add_argument("--cv-folds", help="Number of cross-validation folds", type=int, default=5)
    parser.add_argument("--output-file", help="name of the output file", type=str)
    return parser.parse_args()


def test_reproducibility(df: pd.DataFrame)-> None:
    model_stability_arr = []
    for i in range(10):
        rmse_g_i = get_g_rmse(df, k=args.cv_folds, cv_rs=cv_rs, cb_rs=cb_rs, shuffle_rs=shuffle_rs)
        model_stability_arr.append(rmse_g_i)
    print(model_stability_arr)
    assert len(set(model_stability_arr)) == 1


if __name__ == "__main__":
    args = cli_args()
    if args.download:
        download_kaggle_data(download_path)
        logger.info("Data downloaded successfully.")
    df, cat_features, cont_features, feature_names = load_data(download_path, args.sample_size)


    rmse_results = np.array([])
    for i in range(args.g_count):
        if args.randomize_cv_rs is True:
            cv_rs = i
        if args.randomize_cb_rs is True:
            cb_rs = i
        if args.randomize_shuffle_rs is True:
            shuffle_rs = i
        logger.info(f"Cross-validation random state: {cv_rs}, CatBoostRegressor random state: {cb_rs},"
                    f" Shuffle random state: {shuffle_rs}, g: {i}")
        rmse_g_i = get_g_rmse(df, k=args.cv_folds, cv_rs=cv_rs, cb_rs=cb_rs, shuffle_rs=shuffle_rs)
        rmse_results = np.append(rmse_results, rmse_g_i)
    if not os.path.exists("results"):
        os.makedirs("results")
    g_count_array = np.arange(args.g_count)
    result_arr = np.column_stack((g_count_array, rmse_results))
    os.makedirs("results", exist_ok=True)
    np.savetxt(f"results/{args.output_file}.csv", result_arr, delimiter=",")


