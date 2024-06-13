import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl


def plot_roc_curve(folder, scores, tags):
    """
    Plot ROC curve
    :param folder: The folder to save the roc curve
    :param scores: The scores of the model
    :param tags: The tags of the model
    """

    name = folder.split("/")[-1]
    fpr, tpr, thresholds = roc_curve(tags, scores)

    # Compute AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random guessing)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=15)
    plt.ylabel('True Positive Rate (TPR)', fontsize=15)
    plt.title(f"{name}", fontsize=15, fontweight="bold")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{folder}/roc_curve.png")
    plt.show()


def save_coefficients_to_csv(model, feature_names, filename):
    """
    Save coefficients to a CSV file
    :param model: The model
    :param feature_names: The feature names
    :param filename: The filename
    """
    # Extract coefficients and intercept
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    df = pd.DataFrame({
        'Name': feature_names,
        'Coefficient': coefficients
    })
    intercept_row = pd.DataFrame({'Name': ['Intercept'], 'Coefficient': [intercept]})
    df = pd.concat([intercept_row, df], ignore_index=True)
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

    meta = meta["Tag"]

    tag = meta

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

        # Create logistic regression model
        model = LogisticRegression()

        # Fit the model
        model.fit(to_learn, tag)

        # Predict probabilities for test set
        y_scores = model.predict_proba(to_learn)[:, 1]

        plot_roc_curve(folder, y_scores, tag)
        save_coefficients_to_csv(model, to_learn.columns, f"{folder}/coefficients.csv")

    else:
        print("g__Roseburia or g__Bifidobacterium or f__Aspergillaceae not found in the data.")


if __name__ == '__main__':
    main()
