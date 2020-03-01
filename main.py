
# %%


import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%


def corrSelectFeatures(dataset, size_of_delet_corelation=0.95):
    corr_matrix = dataset.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    select_column = [
        column
        for column in upper.columns
        if any(upper[column] > size_of_delet_corelation)
    ]
    dataset_out = dataset.drop(dataset[select_column], axis=1)
    return dataset_out


def vifSelectFeatures(dataset, thresh=100.0):
    variables = list(range(dataset.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [
            variance_inflation_factor(dataset.iloc[:, variables].values, ix)
            for ix in range(dataset.iloc[:, variables].shape[1])
        ]
        maxloc = vif.index(max(vif))

        if max(vif) > thresh:
            del variables[maxloc]
            dropped = True
    dataset_out = dataset.iloc[:, variables]
    return dataset_out


classifiers = [
    AdaBoostClassifier(),
    DecisionTreeClassifier(max_depth=5),
    ExtraTreesClassifier(n_estimators=5, criterion="entropy", max_features=2),
    GaussianNB(),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    KNeighborsClassifier(),
    LinearDiscriminantAnalysis(),
    LogisticRegression(
        penalty="l1", dual=False, max_iter=110, solver="liblinear", multi_class="auto"
    ),
    MLPClassifier(alpha=1, max_iter=1000),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="sigmoid", gamma=2),
    SVC(kernel="rbf", gamma=2, C=1),
    QuadraticDiscriminantAnalysis(),
]


classifiersNames = [
    "AdaBoost",
    "Decision Tree",
    "Extra Trees",
    "Naive Bayes",
    "Gaussian Process",
    "Nearest Neighbors",
    "Linear Discriminant Analysis",
    "Logistic Regression",
    "Neural Net",
    "Random Forest",
    "SVM Sigmoid",
    "SVM Linear ",
    "SVM RBF",
    "QDA",
]


featureSelectionName = [
    "cor",
    "vif",
    "ga"
]


def initialPopulation(df, count_populations):
    print("start initial population")
    populations = []
    for chromosom in range(count_populations):
        populations.append(np.random.randint(2, size=len(df.columns)))
    print("end initial population")
    return populations


def fitnessEvaluation():
    print("fitnes evaulation")


def reproduction():
    print("reproduction")


def selection():
    print("selection:")
    fitnessEvaluation()
    reproduction()


def mutation():
    print("mutation")


def crossOver():
    print("crossOver")


def geneticOperations():
    print("genetic operation:")
    mutation()
    crossOver()


def gaSelectFeatures(df, count_populations, count_generations):
    print("startGA")
    population = initialPopulation(df, count_populations)
    print(population)
    for actualGenerations in range(count_generations):
        print(actualGenerations)
        selection()
        geneticOperations()


# %%
test_dataframe = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)
test_dataframe.describe()
test_dataframe = test_dataframe.replace("virginica", 3)
test_dataframe = test_dataframe.replace("versicolor", 2)
test_dataframe = test_dataframe.replace("setosa", 1)
test_dataframe["species"] = test_dataframe["species"].astype(int)
print(test_dataframe.head())

# %%

count_of_generations = 5
count_of_populations = 5
gaSelectFeatures(
    test_dataframe,
    count_of_populations,
    count_of_generations
)


# %%
