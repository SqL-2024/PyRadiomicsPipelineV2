from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from mrmr import mrmr_classif
# pip install git+https://github.com/smazzanti/mrmr
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
# pip install boruta
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.neighbors import KNeighborsClassifier
from itertools import cycle
import sklearn_relief as relief
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
import os

np.random.seed(12345)


def chi2_test_selector(feature, label, save_dir, kwargs):
    """

    :param feature:
    :param label:
    :return:
    """
    # print('please input the number of feature to keep:')
    # num_feature = input()
    # ValueError: Input X must be non-negative.
    if 'n_feature' in kwargs.keys():
        num_feature = kwargs['n_feature']
    elif 'p_feature' in kwargs.keys():
        num_feature = int(kwargs['p_feature'] * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting
    scaler = MinMaxScaler()
    new_feature = scaler.fit_transform(feature)
    selector = SelectKBest(chi2, k=num_feature).fit(new_feature, label)
    # chi2_statistics,p_value = chi2(new_feature,label)
    support = selector.get_support()
    return support


def anova_f_value_selector(feature, label, save_dir, kwargs):
    """

    :param feature:
    :param label:
    :param kwargs:
    :return:
    """
    if 'n_feature' in kwargs.keys():
        num_feature = kwargs['n_feature']
    elif 'p_feature' in kwargs.keys():
        num_feature = int(kwargs['p_feature'] * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting

    # print('please input the number of feature to keep:')
    # num_feature = input()
    selector = SelectKBest(f_classif, k=num_feature).fit(feature, label)
    support = selector.get_support()
    return support


def mutual_information_selector(feature, label, save_dir, kwargs):
    """

    :param feature:
    :param label:
    :param kwargs:
    :return:
    """
    if 'n_feature' in kwargs.keys():
        num_feature = kwargs['n_feature']
    elif 'p_feature' in kwargs.keys():
        num_feature = int(kwargs['p_feature'] * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting

    selector = SelectKBest(mutual_info_classif, k=num_feature).fit(feature, label)
    support = selector.get_support()
    return support


def mRMR_selector(feature, label, save_dir, kwargs):
    """

    :param feature:
    :param label:
    :param kwargs:
    :return:
    """
    if 'n_feature' in kwargs.keys():
        num_feature = kwargs['n_feature']
    elif 'p_feature' in kwargs.keys():
        num_feature = int(kwargs['p_feature'] * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting

    selected_features = mrmr_classif(feature, label, K=num_feature)
    # convert the feature index to support
    support = np.array([False] * feature.shape[1])
    support[selected_features] = True
    return support


def refief_selector(feature, label, save_dir, kwargs):
    """

    :param feature:
    :param label:
    :param kwargs:
    :return:
    """
    if 'n_feature' in kwargs.keys():
        num_feature = kwargs['n_feature']
    elif 'p_feature' in kwargs.keys():
        num_feature = int(kwargs['p_feature'] * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting

    selector = relief.Relief().fit(feature, label)
    weight = selector.w_
    weight_sorted = np.flip(np.sort(selector.w_))
    weight_threshold = weight_sorted[num_feature]
    support = weight > weight_threshold
    return support


def RFE_selector(feature, label, save_dir, kwargs):
    """
     recursive feature elimination  with automatic tuning of the number of features selected with cross-validation
    :param feature:
    :param label:
    :param kwargs: the classifier should have `coef_` or `feature_importances_` attribute.
    :return:
    """
    classifier = SVC(kernel="linear")
    min_features_to_select = 1
    rfecv = RFECV(classifier, step=1, cv=StratifiedKFold(5), scoring='accuracy',
                  min_features_to_select=min_features_to_select)
    rfecv.fit(feature, label)
    # save the cross-validation scores for plot self
    print("Optimal number of features : %d" % rfecv.n_features_)
    # plot number of features vs. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(min_features_to_select,
                   len(rfecv.grid_scores_) + min_features_to_select),
             rfecv.grid_scores_)

    plt.axvline(x=np.argmax(rfecv.grid_scores_), linestyle='dashed', c='black', lw=2)
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'recursive_feature_elimination_process.png'))
    support = rfecv.get_support()
    return support


def lasso_selector(feature, label, save_dir, kwargs):
    """
    #Todo
    :param feature:
    :param label:
    :return:
    """
    feature_name = kwargs['feature_name']
    if feature_name is None:
        feature_name = ['feature_{}'.format(i) for i in range(feature.shape[1])]
    classifier = LassoCV(cv=10, random_state=32).fit(feature, label)
    _, coef_path, _ = classifier.path(feature, label)
    alphas = classifier.alphas_

    # plot crierion of cv
    criterion_all = classifier.mse_path_
    criterion_min = criterion_all.min(axis=1)
    criterion_max = criterion_all.max(axis=1)
    criterion_mean = criterion_all.mean(axis=1)
    log_alphas_lasso = np.log10(alphas)
    log_alpha = np.log10(classifier.alpha_)
    fig, ax = plt.subplots(figsize=(8, 6))
    yerr = [np.subtract(criterion_mean, criterion_min), np.subtract(criterion_max, criterion_mean)]
    ax.errorbar(log_alphas_lasso, criterion_mean, yerr=yerr, capsize=5, c='black', linewidth=1)
    ax.scatter(log_alphas_lasso, criterion_mean, c='red', marker='o')
    # ymin, ymax = plt.ylim()
    ax.axvline(x=log_alpha, linestyle='dashed', c='red')
    ax.set_xlabel('Log Lambda')
    ax.set_ylabel('Criterion')
    plt.savefig(os.path.join(save_dir, 'lasso_loss_plot.png'))

    # plt.show()
    # plot coefficient
    plt.figure(figsize=(8, 6))
    # ymin, ymax = plt.ylim()
    for i in range(coef_path.shape[0]):
        plt.plot(log_alphas_lasso, coef_path[i, :])
    plt.axvline(x=log_alpha, linestyle='dashed', c='red')
    plt.xlabel('Log Lambda')
    plt.ylabel('Coefficient')
    plt.savefig(os.path.join(save_dir, 'lasso_coefficient_plot.png'))
    # plt.show()
    alpha_index = np.where(alphas == classifier.alpha_)[0][0]
    coefficient = coef_path[:, alpha_index]
    coef_df = pd.DataFrame(coefficient[np.newaxis], columns=feature_name)
    coef_df.to_excel(os.path.join(save_dir, 'lasso_coefficient.xlsx'))
    support = coefficient != 0
    # new_name =
    coef_no_zero = coefficient[support]
    coef_no_zero_name = feature_name[support]
    x_ticks = coef_no_zero_name
    heights = coef_no_zero
    x_pos = [i for i, _ in enumerate(x_ticks)]
    plt.figure()
    plt.bar(x_pos, heights, color='gold')
    plt.xlabel("feature_name")
    plt.ylabel("importance")
    plt.title("feature importance after lasso")
    plt.xticks(x_pos, x_ticks)
    plt.savefig(os.path.join(save_dir, 'lasso_feature_importance.png'))
    return support


def RF_selector(feature, label, save_dir, kwargs):
    """

    :param feature:
    :param label:
    :param kwargs:
    :return:
    """
    if 'n_feature' in kwargs.keys():
        num_feature = kwargs['n_feature']
    elif 'p_feature' in kwargs.keys():
        num_feature = int(kwargs['p_feature'] * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting
    feature_name = kwargs['feature_name']
    forest = RandomForestClassifier(random_state=0)
    forest.fit(feature, label)
    importances = forest.feature_importances_
    std = np.std([
        tree.feature_importances_ for tree in forest.estimators_], axis=0)
    feature_name = None
    if feature_name is None:
        feature_name = ['feature_{}'.format(i) for i in range(feature.shape[1])]

    forest_importances = pd.Series(importances, index=feature_name)  # for plot
    forest_importances.to_excel(os.path.join(save_dir, 'feature_importance_with_random_forest.xlsx'))
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance_with_random_forest.png'))

    importances_sorted = np.sort(importances)[::-1]  # sorted max-> min
    importances_threshold = importances_sorted[num_feature]  # get the threshold for filtering
    support = importances >= importances_threshold
    return support


def SFS_selector(feature, label, save_dir, kwargs):
    """
    :param feature:
    :param label:
    :return:
    """
    if 'n_feature' in kwargs.keys():
        num_feature = kwargs['n_feature']
    elif 'p_feature' in kwargs.keys():
        num_feature = int(kwargs['p_feature'] * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting
    direction = kwargs['direction']
    if direction == None:
        direction = 'forward'
    classifier_tag = kwargs['classifier']
    if classifier_tag == 'lasso':
        classifier = LassoCV()
    elif classifier_tag == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=3)
    elif classifier_tag == 'svm':
        classifier = SVC()
    elif classifier_tag == 'random_forest':
        classifier = RandomForestClassifier(random_state=0)

    sfs_selector = SequentialFeatureSelector(classifier, n_features_to_select=num_feature, direction=direction).fit(
        feature, label)
    support = sfs_selector.get_support()

    return support


def Boruta_selector(feature, label, save_dir, kwargs):
    """

    :param feature:
    :param label:
    :return:
    """
    forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
    feat_selector.fit(feature, label)
    support = feat_selector.support_
    return support




if __name__ == '__main__':
    # test
    import pandas as pd

    train = pd.read_csv('G:/GIST_preprocessed/MLGIST/train_set_varianceAndMRMRFiltered.csv')
    y = train['label'].values
    data = train.drop(columns='label')
    x = data.values
    print(x.shape)
