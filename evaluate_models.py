# --------------------------------------------------------------------------------------------------
import pprint
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def report(
    y_test, y_test_pred, y_train=None, y_train_pred=None, model=None, term=False
):
    line = "-" * 15

    # model infos
    if model is not None:
        type_ = str(type(model))

        if "DecisionTreeClassifier" in type_:
            tree_name = "Decision Tree"
            tree_depth = model.tree_.max_depth
            tree_nodes = model.tree_.node_count
            txt_avg = ""

            # how deep
            print(
                f'"{tree_name}" has {tree_nodes} nodes {txt_avg} with \
                    maximum depth {tree_depth} {txt_avg}.'
            )

        elif "RandomForestClassifier" in type_:
            print("current parameter:")
            pprint.PrettyPrinter(width=20).pprint(model.get_params())

            n_nodes = []
            max_depths = []

            for ind_tree in model.estimators_:
                n_nodes.append(ind_tree.tree_.node_count)
                max_depths.append(ind_tree.tree_.max_depth)

            tree_name = "Random Forest"
            tree_depth = int(np.mean(max_depths))
            tree_nodes = int(np.mean(n_nodes))
            txt_avg = "on average"

            # how deep
            print(
                f'"{tree_name}" has {tree_nodes} nodes {txt_avg} with \
                    maximum depth {tree_depth} {txt_avg}.'
            )

        elif "RandomizedSearchCV" in type_:
            best_model = model.best_estimator_
            bm_type = str(type(best_model))

            print("best parameter:")
            pprint.PrettyPrinter(width=20).pprint(model.best_params_)

            # how deep?
            n_nodes = []
            max_depths = []

            for ind_tree in best_model.estimators_:
                n_nodes.append(ind_tree.tree_.node_count)
                max_depths.append(ind_tree.tree_.max_depth)

            tree_name = bm_type.split(".")[-1].split("'")[0]
            tree_depth = int(np.mean(max_depths))
            tree_nodes = int(np.mean(n_nodes))
            txt_avg = "on average"

            # how deep
            print(
                f'"{tree_name}" has {tree_nodes} nodes {txt_avg} with \
                    maximum depth {tree_depth} {txt_avg}.'
            )

    # confusion matrix
    if y_train is not None:
        cfmat_train = pd.crosstab(
            y_train, y_train_pred, rownames=["Actual"], colnames=["Predicted"]
        )
    cfmat_test = pd.crosstab(
        y_test, y_test_pred, rownames=["Actual"], colnames=["Predicted"]
    )
    if term:
        if y_train is not None:
            print(line + " confusion matrix for Train " + line)
            print(cfmat_train)
        print(line + " confusion matrix for Test " + line)
        print(cfmat_test)
    else:
        cmap = sns.light_palette("seagreen", as_cmap=True)

        # Plot confusion matrices
        plt.figure(figsize=(10, 4))
        if y_train is not None:
            plt.subplot(1, 2, 1)
            sns.heatmap(
                confusion_matrix(y_train, y_train_pred), annot=True, cmap=cmap, fmt="g"
            )
            plt.title("Confusion Matrix for Train")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

        plt.subplot(1, 2, 2)
        sns.heatmap(
            confusion_matrix(y_test, y_test_pred), annot=True, cmap=cmap, fmt="g"
        )
        plt.title("Confusion Matrix for Test")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        plt.show()

    # ConfusionMatrixDisplay(y_train, y_train_pred)

    # classification report
    if y_train is not None:
        print(line + " classification report for Train " + line)
        print(classification_report(y_train, y_train_pred, digits=3))
    print(line + " classification report for Test " + line)
    print(classification_report(y_test, y_test_pred, digits=3))
