import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, DotProduct, RationalQuadratic
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from scipy.stats import randint
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
import os
import pandas as pd


def feature_classification_process(classifier_tags, train_frame, validation_frame, save_dir, random_state=42,
                                   scoring='accuracy'):
    """
    for two-class
    :param classifier_tags:
    :param train_frame:
    :param validation_frame:
    :param save_dir:
    :param save_model:
    :param scoring:
    :param random_state:
    :return:
    """

    classifier_dict = {
        0: 'LogsticRegression',
        1: 'GaussianProcessClassification',
        2: 'SVM',
        3: 'SGDClassifier',
        4: 'KNeighborsNearest',
        5: 'DecisionTree',
        6: 'RandomForest',
        7: 'Adaboost',
        8: 'Naive Bayes',
        9: 'Quadratic Discriminant Analysis'
    }

    label_train = train_frame['label'].values
    feature_train = train_frame.drop(['case_ids', 'label'], axis=1).values
    num_samples, num_feature = feature_train.shape
    case_ids_train = train_frame['case_ids'].values

    label_validation = validation_frame['label'].values
    case_ids_test = validation_frame['case_ids'].values
    feature_validation = validation_frame.drop(['case_ids', 'label'], axis=1).values

    train_probs = []
    validation_probs = []
    # train_preds = []
    # validation_preds = []
    columns = []
    method_save_dir = os.path.join(save_dir, 'train_models')
    os.makedirs(method_save_dir, exist_ok=True)
    for classifier_tag in classifier_tags:
        classification_method = classifier_dict[classifier_tag]
        columns.append(classification_method)
        file_name = os.path.join(method_save_dir, classification_method + '_trained.sav')
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                clf_best = pickle.load(f)
                print('load trained model:', classification_method)
        else:
            # pickle.dump(clf_best, open(file_name, 'wb'))
            if classifier_tag == 0:  # logistic regression
                # if os.path.exists(os.path.join(method_save_dir,'{}_trained.'))
                clf_best = LogisticRegressionCV(cv=10, random_state=random_state)
                clf_best.fit(feature_train, label_train)
            elif classifier_tag == 1:  # Gaussian process classification

                # define grid
                # example:https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes
                model = GaussianProcessClassifier(random_state=random_state)
                param_grid = dict()
                param_grid['kernel'] = [1 * RBF(), 1 * DotProduct(), 1 * Matern(), 1 * RationalQuadratic(),
                                        1 * WhiteKernel()]
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
                search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=1)
                results = search.fit(feature_train, label_train)
                # summarize best
                print('Best Mean Accuracy: %.3f' % results.best_score_)
                print('Best Config: %s' % results.best_params_)
                # summarize all
                means = results.cv_results_['mean_test_score']
                params = results.cv_results_['params']
                for mean, param in zip(means, params):
                    print(">%.3f with: %r" % (mean, param))
                clf_best = GaussianProcessClassifier(kernel=results.best_params_['kernel'],
                                                     random_state=random_state)  # the best parameters
                clf_best.fit(feature_train, label_train)
            elif classifier_tag == 2:  # svm
                # Set the parameters by cross-validation
                model = svm.SVC(probability=True, random_state=random_state)
                param_grid = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000]},
                              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                              {'kernel': ['poly'], 'C': [1, 10, 100, 1000]},
                              {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000]}
                              ]
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
                search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=1)
                results = search.fit(feature_train, label_train)
                print('Best Mean Accuracy: %.3f' % results.best_score_)
                print('Best Config: %s' % results.best_params_)
                # summarize all
                means = results.cv_results_['mean_test_score']
                params = results.cv_results_['params']
                for mean, param in zip(means, params):
                    print(">%.3f with: %r" % (mean, param))
                # clf = svm.SVC()
                clf_best = svm.SVC(kernel=results.best_params_['kernel'], C=results.best_params_['C'], probability=True,
                                   random_state=random_state)
                clf_best.fit(feature_train, label_train)
            elif classifier_tag == 3:  # stochastic gradient descent learning
                param_grid = {
                    'loss': ['log', 'modified_huber', 'perceptron'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'average': [True, False],
                    'l1_ratio': np.linspace(0, 1, num=10),
                    'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
                model = SGDClassifier(fit_intercept=True, random_state=random_state)
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
                search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=1)
                results = search.fit(feature_train, label_train)
                print('Best Mean Accuracy: %.3f' % results.best_score_)
                print('Best Config: %s' % results.best_params_)
                means = results.cv_results_['mean_test_score']
                params = results.cv_results_['params']
                for mean, param in zip(means, params):
                    print(">%.3f with: %r" % (mean, param))

                clf_best = SGDClassifier(loss=results.best_params_['loss'],
                                         penalty=results.best_params_['penalty'],
                                         l1_ratio=results.best_params_['l1_ratio'],
                                         alpha=results.best_params_['alpha'],
                                         average=results.best_params_['average'],
                                         fit_intercept=True, random_state=random_state)
                clf_best.fit(feature_train, label_train)

            elif classifier_tag == 4:  # neighbor
                param_grid = {
                    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
                model = KNeighborsClassifier()
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
                search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=1)
                results = search.fit(feature_train, label_train)
                print('Best Mean Accuracy: %.3f' % results.best_score_)
                print('Best Config: %s' % results.best_params_)
                means = results.cv_results_['mean_test_score']
                params = results.cv_results_['params']
                for mean, param in zip(means, params):
                    print(">%.3f with: %r" % (mean, param))

                clf_best = KNeighborsClassifier(n_neighbors=results.best_params_['n_neighbors'],
                                                weights=results.best_params_['weights'],
                                                algorithm=results.best_params_['algorithm'])
                clf_best.fit(feature_train, label_train)

            elif classifier_tag == 5:  # DecisionTree
                # all_samples = feature_train.shape[0]
                param_grid = {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'min_samples_split': range(2, num_samples, 5)}
                model = tree.DecisionTreeClassifier(random_state=random_state)
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
                search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=1)
                results = search.fit(feature_train, label_train)
                print('Best Mean Accuracy: %.3f' % results.best_score_)
                print('Best Config: %s' % results.best_params_)
                means = results.cv_results_['mean_test_score']
                params = results.cv_results_['params']
                for mean, param in zip(means, params):
                    print(">%.3f with: %r" % (mean, param))

                clf_best = tree.DecisionTreeClassifier(criterion=results.best_params_['criterion'],
                                                       min_samples_split=results.best_params_['min_samples_split'],
                                                       splitter=results.best_params_['splitter'],
                                                       random_state=random_state)
                clf_best.fit(feature_train, label_train)


            elif classifier_tag == 6:  # RandomForest
                model = RandomForestClassifier(random_state=random_state)
                param_grid = {
                    'n_estimators': np.arange(20, 100, 5),
                    "max_depth": np.linspace(1, 110, num=10),
                    "max_features": ['sqrt', 'log2'],
                    "min_samples_split": [2, 5, 10],
                    "bootstrap": [True, False]}
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
                search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=1)
                results = search.fit(feature_train, label_train)
                print('Best Mean Accuracy: %.3f' % results.best_score_)
                print('Best Config: %s' % results.best_params_)
                means = results.cv_results_['mean_test_score']
                params = results.cv_results_['params']
                for mean, param in zip(means, params):
                    print(">%.3f with: %r" % (mean, param))

                clf_best = RandomForestClassifier(n_estimators=results.best_params_['n_estimators'],
                                                  max_depth=results.best_params_['max_depth'],
                                                  max_features=results.best_params_['max_features'],
                                                  min_samples_split=results.best_params_['min_samples_split'],
                                                  bootstrap=results.best_params_['bootstrap'],
                                                  random_state=random_state)
                clf_best.fit(feature_train, label_train)


            elif classifier_tag == 7:  # adaboost
                model = AdaBoostClassifier(random_state=random_state)
                param_grid = {
                    "n_estimators": np.arange(5, 210, 20),
                    "algorithm": ['SAMME.R', "SAMME"],
                    "learning_rate": np.linspace(0.01, 1, 10)
                }
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
                search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=1)
                results = search.fit(feature_train, label_train)
                print('Best Mean Accuracy: %.3f' % results.best_score_)
                print('Best Config: %s' % results.best_params_)
                means = results.cv_results_['mean_test_score']
                params = results.cv_results_['params']
                for mean, param in zip(means, params):
                    print(">%.3f with: %r" % (mean, param))

                clf_best = AdaBoostClassifier(n_estimators=results.best_params_['n_estimators'],
                                              algorithm=results.best_params_['algorithm'],
                                              learning_rate=results.best_params_['learning_rate'],
                                              random_state=random_state)
                clf_best.fit(feature_train, label_train)
            elif classifier_tag == 8:  # Naive Bayes
                clf_best = GaussianNB()
                clf_best.fit(feature_train, label_train)
            elif classifier_tag == 9:
                clf_best = QuadraticDiscriminantAnalysis()
                clf_best.fit(feature_train, label_train)

            file_name = os.path.join(method_save_dir, classification_method + '_trained.sav')
            pickle.dump(clf_best, open(file_name, 'wb'))
        train_probs.append(clf_best.predict_proba(feature_train)[:, 1])
        validation_probs.append(clf_best.predict_proba(feature_validation)[:, 1])

    train_probs = np.array(train_probs)
    train_probs = pd.DataFrame(train_probs.transpose(), columns=columns)
    train_probs.insert(0, 'case_ids', case_ids_train)
    train_probs.insert(1, 'label', label_train)
    train_probs.to_excel(os.path.join(save_dir, 'train_set_classification_result.xlsx'), index=False)
    validation_probs = np.array(validation_probs)
    validation_probs = pd.DataFrame(validation_probs.transpose(), columns=columns)
    validation_probs.insert(0, 'case_ids', case_ids_test)
    validation_probs.insert(1, 'label', label_validation)
    validation_probs.to_excel(os.path.join(save_dir, 'test_set_classification_result.xlsx'), index=False)
    return train_probs, validation_probs
