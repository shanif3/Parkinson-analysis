import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_curve, auc



def plot_roc(folder, pred_probabilities, true_labels):
    name = folder.split("/")[-1]
    fpr, tpr, _ = roc_curve(true_labels, pred_probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{name}", fontsize=15, fontweight="bold")
    plt.legend(loc='lower right')
    plt.savefig(f'{folder}/roc_curve_update.png')


def save_coefficients_to_csv(mean_coefficients, feature_names, filename):
    """
    Save coefficients to a CSV file
    :param mean_coefficients: Averaged coefficients from the model
    :param feature_names: The feature names
    :param filename: The filename
    """
    df = pd.DataFrame({
        'Name': feature_names,
        'Coefficient': mean_coefficients
    })
    df.to_csv(filename, index=False)


def main():
    folder = "/home/shanif3/Codes/MIPMLP/data_to_compare/Parkinson/Parkinson-git/Validation/16S"
    mpl.rc('font', family='Times New Roman')

    s16 = pd.read_csv(f"{folder}/processed_afterMIPMLP.csv", index_col=0)
    meta = pd.read_csv(f"{folder}/tag.csv", index_col=0)

    common = list(meta.index.intersection(s16.index))
    meta = meta.loc[common]
    s16 = s16.loc[common]

    meta = meta.dropna()
    s16 = s16.loc[meta.index]

    tag = meta["Tag"]
    substrings = [
        "f__Lachnospiraceae;g__Roseburia",
        "f__Bifidobacteriaceae;g__Bifidobacterium",
        "d__Eukaryota;p__Ascomycota;c__Eurotiomycetes;o__Eurotiales;f__Aspergillaceae"
    ]

    relevant_bac = [col for col in s16.columns if any(substring in col for substring in substrings)]
    if relevant_bac:
        to_learn = s16[relevant_bac]

        common = list(to_learn.index.intersection(tag.index))
        tag = tag.loc[common]
        tag = tag.astype(float)
        to_learn = to_learn.loc[common]

        skf = StratifiedKFold(n_splits=5)
        model = LogisticRegression(max_iter=10000)

        true_labels = []
        pred_probabilities = []

        for train_index, test_index in skf.split(to_learn, tag):
            X_train, y_train = to_learn.iloc[train_index], tag.iloc[train_index]
            X_test, y_test = to_learn.iloc[test_index], tag.iloc[test_index]

            loo = LeaveOneOut()

            for loo_train_index, loo_test_index in loo.split(X_train):
                X_loo_train, y_loo_train = X_train.iloc[loo_train_index], y_train.iloc[loo_train_index]
                model.fit(X_loo_train, y_loo_train)



            model.fit(X_train, y_train)
            true_labels.extend(y_test)
            y_scores = model.predict_proba(X_test)[:, 1]
            pred_probabilities.extend(y_scores)

        mean_coefficients = model.coef_[0]
        feature_names = to_learn.columns.tolist()
        save_coefficients_to_csv(mean_coefficients, feature_names, f"{folder}/coefficients.csv")
        plot_roc(folder,pred_probabilities, true_labels)



    else:
        print("g__Roseburia or g__Bifidobacterium or f__Aspergillaceae not found in the data.")


if __name__ == '__main__':
    main()
