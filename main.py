
# %%

def initialFeatureVector():
    print("initial feature vector")

def initialPopulation():
    print("initial population")


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

def featureSelectionGA(generations):
    print("startGA")
    initialFeatureVector()
    initialPopulation()
    for actualGenerations in range(generations):
        selection()
        geneticOperations()




featureSelectionGA(10)


# %%
