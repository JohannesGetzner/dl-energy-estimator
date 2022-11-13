import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def plot_estimate_vs_ground_truth(y, y_hat):
    plt.figure(figsize=(4, 3))
    g = sns.regplot(x=y, y=y_hat, x_ci=None, ci=None)
    min_x = min(min(y), min(y_hat))
    max_x = max(max(y), max(y_hat))
    if isinstance(min_x, np.ndarray): min_x = min_x[0]
    if isinstance(max_x, np.ndarray): max_x = max_x[0]
    g.plot([min_x, max_x], [min_x, max_x], transform=g.transData, linestyle="--", color="#f032e6")
    g.set_xlabel("Ground Truth")
    g.set_ylabel("Prediction")
    custom_lines = [
        plt.Line2D([], [], color="#EAEAF2", marker='o', markersize=8, markerfacecolor="#4C72B0"),
        plt.Line2D([0], [0], color="#4C72B0", lw=2, linestyle="--"),
        plt.Line2D([0], [0], color="#f032e6", lw=2, linestyle="--"),
    ]
    plt.legend(custom_lines, ["predictions", "regressor", "ideal"], loc="upper left")
    plt.show()


def apply_data_transforms(dfs, transformers_dict):
    if transformers_dict["x_preprocessors"]:
        for p in transformers_dict["x_preprocessors"]:
            dfs['x_train'] = p.fit_transform(dfs['x_train'])
            dfs['x_val'] = p.transform(dfs['x_val'])
            dfs['x_test'] = p.transform(dfs['x_test'])
    if transformers_dict["y_preprocessor"] is not None:
        dfs['y_train'] = transformers_dict["y_preprocessor"].fit_transform(dfs['y_train'].to_numpy().reshape(-1, 1))
        dfs['y_val'] = transformers_dict["y_preprocessor"].transform(dfs['y_val'].to_numpy().reshape(-1, 1))
        dfs['y_test'] = transformers_dict["y_preprocessor"].transform(dfs['y_test'].to_numpy().reshape(-1, 1))
    return dfs, transformers_dict


def compute_log_transformed_features(df, features_to_transform):
    # compute the log transformed features
    data_w_log_features = df.copy()
    new_cols = []
    for col in features_to_transform:
        data_w_log_features[f"log_{col}"] = np.log1p(data_w_log_features[col])
        new_cols.append(f"log_{col}")
    print("New Columns: ", features_to_transform + new_cols)
    return data_w_log_features, features_to_transform + new_cols


def fit_model(model, x_train, y_train, x_val, y_val, plot_results=False):
    r2_scores = cross_val_score(
        model, x_train, y_train.ravel(), cv=2, n_jobs=-1, scoring="r2"
    )
    mses = cross_val_score(
        model, x_train, y_train.ravel(), cv=2, n_jobs=-1, scoring="neg_mean_squared_error"
    )
    model = model.fit(x_train, y_train.ravel())
    val_score = model.score(x_val, y_val.ravel())
    val_mse = mean_squared_error(y_val.ravel(), model.predict(x_val))

    print("-" * 20)
    print("Average R2 Cross-Validation Score: {:.3f} (± {:.3f})".format(np.average(r2_scores), np.std(r2_scores)))
    print("Average MSE Cross-Validation: {:.3e} (± {:.3e})".format(np.average(mses), np.std(mses)))
    print("Validation R2 Score: {:.3f}".format(val_score))
    print("Validation MSE: {:.3e}".format(val_mse))
    if plot_results:
        plot_estimate_vs_ground_truth(y_val, model.predict(x_val))
    return model, val_score, val_mse


def split_data_set(df, feature_names, SEED):
    train, val, test = np.split(df.sample(frac=1, random_state=SEED), [int(.7 * len(df)), int(.9 * len(df))])
    dfs = {
        "x_train": train[feature_names],
        "y_train": train['cpu_energy'],
        "x_val": val[feature_names],
        "y_val": val['cpu_energy'],
        "x_test": test[feature_names],
        "y_test": test['cpu_energy']
    }
    return dfs


def test_model(model, x_test, y_test, plot_results=False):
    y_hat = model.predict(x_test).reshape(-1, 1)
    test_score = model.score(x_test, y_test.ravel())
    test_mse = mean_squared_error(y_test.ravel(), y_hat)
    print("Test R2 Score: {:.3f}".format(test_score))
    print("Test MSE: {:.3e}".format(test_mse))
    if plot_results:
        plot_estimate_vs_ground_truth(y_test, y_hat)
    return y_hat, test_score, test_mse