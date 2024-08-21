import pickle as pk
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

def supress_exceptions(f):
    def decorated(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except:
            pass
    return decorated

def single_decision_tree(X_train, X_test, y_train, y_test, persist=False, print_all_metrics=False):
    clf = DecisionTreeClassifier(random_state=42)
    gscv = GridSearchCV(clf, param_grid={ 'criterion': ['gini', 'entropy', 'log_loss'] })
    start_time = time.time()
    gscv.fit(X_train, y_train)
    end_time = time.time()
    if print_all_metrics:
        print("Tempo de treino em segundos: ", end_time - start_time)
        print_metrics(gscv, X_test, y_test)
    if persist:
        with open('./models/decision_tree.sav', 'wb') as pickle_file:
            pk.dump(gscv, pickle_file)
    return gscv

def random_forest(X_train, X_test, y_train, y_test, persist=False, print_all_metrics=False):
    clf = RandomForestClassifier(random_state=42)
    gscv = GridSearchCV(clf, param_grid={ 'criterion': ['gini', 'entropy', 'log_loss'] })
    start_time = time.time()
    gscv.fit(X_train, y_train)
    end_time = time.time()
    if print_all_metrics:
        print("Tempo de treino em segundos: ", end_time - start_time)
        print_metrics(gscv, X_test, y_test)
    if persist:
        with open('./models/random_forest.sav', 'wb') as pickle_file:
            pk.dump(gscv, pickle_file)
    return gscv

def xgboost_model(X_train, X_test, y_train, y_test, persist=False, print_all_metrics=False):
    clf = xgb.XGBClassifier(num_class=10)
    gscv = GridSearchCV(clf, param_grid={ 'max_depth': [10, 100, 1000] })
    start_time = time.time()
    gscv.fit(X_train, y_train)
    end_time = time.time()
    if print_all_metrics:
        print("Tempo de treino em segundos: ", end_time - start_time)
        print_metrics(gscv, X_test, y_test)
    if persist:
        with open('./models/xgboost.sav', 'wb') as pickle_file:
            pk.dump(gscv, pickle_file)
    return gscv

@supress_exceptions
def print_metrics(model, X_test, y_test, show_tree=False):
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print("Precisão:", score)
    print("Melhores parametros:", model.best_params_)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    disp.plot()
    plt.show()
    if show_tree:
        plot_tree(model, filled=True)
        plt.title("Decision tree trained on all the MNIST features")
        plt.show()

def get_normalizer(dataset, persist=False):
    scaler = StandardScaler(copy=False)
    standardized_dataset = scaler.fit_transform(dataset)
    if persist:
        with open('./models/normalizer.sav', 'wb') as pickle_file:
            pk.dump(scaler, pickle_file)
    return [scaler, standardized_dataset]

# Carregar o dataset digits
digits = load_digits()
X = digits.data
y = digits.target
scaler, standardized_dataset = get_normalizer(X, persist=True)
X_train, X_test, y_train, y_test = train_test_split(standardized_dataset, y, test_size=0.2, random_state=42)
single_decision_tree(X_train, X_test, y_train, y_test, persist=True, print_all_metrics=True)
random_forest(X_train, X_test, y_train, y_test, persist=True, print_all_metrics=True)
xgboost_model(X_train, X_test, y_train, y_test, persist=True, print_all_metrics=True)
