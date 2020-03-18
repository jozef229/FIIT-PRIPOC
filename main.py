# %%
import urllib.request
import random
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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


estimatorFunction = [
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


estimatorNames = [
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
    "ga",
    "none"
]


def initialPopulation(df, count_of_populations):
    populations = []
    for chromosom in range(count_of_populations):
        populations.append(np.random.randint(2, size=len(df.columns)-1))
    return populations


def fitnessEvaluation(estimator, df, populations, main_value_of_dataset):
    y = df[main_value_of_dataset]
    X = df[list(filter(lambda x: x != main_value_of_dataset, df.columns.tolist()))]
    score = []
    for chromosom in populations:
        score.append(-1.0 * np.mean(cross_val_score(estimator,
                                                    X.iloc[:, chromosom], y, cv=5, scoring="neg_mean_squared_error")))
    return np.array(populations)[np.argsort(np.array(score)), :], max(score), mean(score)


def select(populations, df, count_of_best_chromosome_to_select, count_of_random_chromosome_to_select):
    for random_chromosom in range(count_of_random_chromosome_to_select):
        populations[count_of_best_chromosome_to_select +
                    random_chromosom] = np.random.randint(2, size=len(df.columns)-1)


def mutation(default_populations, chance_of_chromosome_mutation):
    populations = default_populations.copy()
    mutation_populations = []
    for chromosom in populations:
        if random.random() < chance_of_chromosome_mutation:
            for i in range(len(chromosom)):
                if random.random() < 0.5:
                    chromosom[i] = 1 - chromosom[i]
        mutation_populations.append(chromosom)
    return mutation_populations


def crossOver(populations, count_of_children_to_crossover):
    populations
    crossover_populations = []
    count = count_of_children_to_crossover
    if count_of_children_to_crossover > len(populations)//2:
        count = len(populations)//2
    for i in range(count):
        child, chromosom = populations[i], populations[len(populations)-1-i]
        change = False
        for j in range(len(child)):
            if change == False and random.random() < 0.5:
                change = True
            if change == True:
                chromosom[j] = child[j]


def plotScores(max_score, average_score):
    plt.plot(max_score, label='Max score')
    plt.plot(average_score, label='Average score')
    plt.legend()
    plt.ylabel('Scores')
    plt.xlabel('Generation')
    plt.show()


def gaSelectFeatures(estimator, df, count_populations, count_of_generations, count_of_children_to_crossover, count_of_best_chromosome_to_select, count_of_random_chromosome_to_select, chance_of_chromosome_mutation, main_value_of_dataset):
    populations = initialPopulation(df, count_populations)
    maxScore = 0
    maxChromosom = []
    avgScoreArray = []
    maxScoreArray = []
    for actualGenerations in range(count_of_generations):
        print(str(actualGenerations) + " -Generation")
        orderPopulations, actualMaxScore, actualAvgScore = fitnessEvaluation(
            estimator, df, populations, main_value_of_dataset)
        maxScoreArray.append(actualMaxScore)
        avgScoreArray.append(actualAvgScore)
        if maxScore < actualMaxScore:
            maxChromosom = orderPopulations[0]
            maxScore = actualMaxScore
        crossOver(orderPopulations, count_of_children_to_crossover)
        select(orderPopulations, df, count_of_best_chromosome_to_select,
               count_of_random_chromosome_to_select)
        mutation(orderPopulations, chance_of_chromosome_mutation)
    plotScores(maxScoreArray, avgScoreArray)
    return pd.concat([df[main_value_of_dataset], X.iloc[:, maxChromosom]], axis=1, sort=False)


# %%

# Set display
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)

# https://www.kaggle.com/iabhishekofficial/mobile-price-classification#train.csv
test_dataframe = pd.read_csv('Dataset/train.csv')


# %%

model = estimatorFunction[0]
count_of_populations = 100
count_of_generations = 20
count_of_children_to_crossover = 30
count_of_best_chromosome_to_select = 30
count_of_random_chromosome_to_select = 30
chance_of_chromosome_mutation = 0.6
main_value_of_dataset = 'price_range'

# gaSelectFeatures(
#     model,
#     test_dataframe,
#     count_of_populations,
#     count_of_generations,
#     count_of_children_to_crossover,
#     count_of_best_chromosome_to_select,
#     count_of_random_chromosome_to_select,
#     chance_of_chromosome_mutation,
#     main_value_of_dataset
# )


df_stats_model = pd.DataFrame()
for featureSelection in range(len(featureSelectionName)):
    df_data = test_dataframe.copy()
    print()
    print(str(featureSelection) + " FS from " + len(featureSelectionName))
    if featureSelection == 0:
        df_data = corrSelectFeatures(df_data)
    if featureSelection == 1:
        df_data = vifSelectFeatures(df_data)
    if featureSelection == 2:
        df_data = gaSelectFeatures(
            model,
            df_data,
            count_of_populations,
            count_of_generations,
            count_of_children_to_crossover,
            count_of_best_chromosome_to_select,
            count_of_random_chromosome_to_select,
            chance_of_chromosome_mutation,
            main_value_of_dataset
        )
    for estimator in range(len(estimatorNames)):
        print(str(estimator) + "Estimator" + len(estimatorNames))
        df_test = df_data.copy()
        y = test_dataframe[main_value_of_dataset]
        X = df_test[list(
            filter(lambda x: x != main_value_of_dataset, df_test.columns.tolist()))]
        try:
            df_stats_model = df_stats_model.append({
                "Features Selecetion": featureSelectionName[featureSelection],
                "Score": -1.0 * np.mean(cross_val_score(estimatorFunction[estimator], X, y, cv=5, scoring="neg_mean_squared_error")),
                "model": estimatorNames[estimator]
            }, ignore_index=True)
            pass
        except ValueError:
            pass

df_stats_model.to_csv(r'columns_stats_model.csv', index=False, header=True)
cm = sns.light_palette("green", as_cmap=True)
styled = df_stats_model.style.background_gradient(cmap=cm)
styled.to_excel('Documentation/styled_model.xlsx', engine='openpyxl')


# %%
