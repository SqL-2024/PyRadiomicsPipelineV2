import argparse
import numpy as np
from run.run_training import Trainer


# import warnings
#
# warnings.simplefilter('ignore', DeprecationWarning)


def main():
    parser = argparse.ArgumentParser()
    # setting of  basic data
    parser.add_argument("-t", "--data_dir", type=str, required=True,
                        help="path of train set")
    parser.add_argument("-i", "--patient_info", type=str, required=True,
                        help="path of excel file for patient case and label information.")
    parser.add_argument("-s", "--save_dir", type=str, required=True,
                        help="path to save extracted features and classification results.")
    parser.add_argument("-v", "--test_size", type=float, default=0.3,
                        help="split the train set to train and internel validation set.")

    # setting to extract feature
    parser.add_argument("-m", "--modality", default="T1,T2", type=str,
                        help='user-defined dicom modility to extract radiomic features')
    # parser.add_argument("-l", "--label_tag", default=1, type=int,
    #                     help='label of the the mask used to extract features.')
    parser.add_argument("-rt", "--radiomic_feature_types", type=str,
                        default="first_order,shape3d,glcm,glrlm,glszm,ngtdm,gldm",
                        help='the class of feature to extract,the detail information please check the website:'
                             'https://pyradiomics.readthedocs.io/en/latest/features.html')
    parser.add_argument("-f", "--image_filters", type=str, default="original,log_1,log_3,log_5,wavelet",
                        help='the filter was applied to the input image and then extract the features,detail information '
                             'please check the website:https://pyradiomics.readthedocs.io/en/latest/radiomics.html#module-radiomics.imageoperations')

    # setting to preprocess features
    parser.add_argument("-p", "--feature_preprocessing_method", default=0,
                        help='0-Standardization(x-mean)/std,1-Normalization(rescale the value between 0-1)')  #

    # setting for feature selection
    parser.add_argument("-fs", "--feature_selection_methods", type=str,
                        default="1,p_feature,0.5/4,p_feature,0.5/8",
                        # default=""{"1: {"n_feature": 200}, 7: {}}",
                        help='method for feature selection:'
                             '1-filter method based on chi2 test,set the number or percent of features to keep.'
                             'example {"n_feature":300} means the number of remaining features is 200 after this process '
                             'and {"p_feature"：0.5} means half number of feature is kept after this process, the percent shoule be in 0-1'
                             '2-filter method based on f value,set the number or percent of features to keep.'
                             '3-correlation analysis of mutual information,set the number or percent of features to keep.'
                             '4-minimum redundancy maximum relevance,set the number of feature of percent of feature to keep'
                             '5-Relief  algorithm for feature selection,set the number or percent of features to keep,'
                             '6-recursive feature elimination with SVM,could be {} means no settings'
                             '7-Lasso,{}'  #
                             '8-Importance of RandomForest,{}'
                             '9-Sequential Feature Selection based on model,set the number or percent of features to keep,and the '
                             'classifier parameter is the estimator to choose the best features, here including lasso,knn.svm,random_forest'
                             'direction parameter to control whether forward or backward SFS is used,'
                             'for example: {"num_features":20,"classifier":"random_forest","direction":"forward"}'
                             '10-Boruta.'
                        )
    # setting for feature classification
    parser.add_argument("-c", "--classifier", default="0, 1, 2,3,4,5,6,7,8,9", type=str,
                        help='classification method: 0-LogsticRegression,1-GaussianProcessClassifier,'
                             '2-SVM,3-SGDClassifier,4-KNeighborsNearest,5-DecisionTree,'
                             '6-RandomForest,7-Adaboost,8-Naive Bayes,9-QuadraticDiscriminantAnalysis')

    parser.add_argument("-r", "--random_seed", default=123, help='random seed to split the data to train and test set')

    args = parser.parse_args()
    data_dir = args.data_dir
    patient_info = args.patient_info
    test_size = args.test_size
    save_dir = args.save_dir

    modality_for_extraction = [item for item in args.modality.split(',')]  # convert string to list
    print('modality_for_extraction', modality_for_extraction)

    # label_tag = args.label_tag
    radiomic_feature_types = [item for item in args.radiomic_feature_types.split(',')]
    print('radiomic_feature_types', radiomic_feature_types)
    image_filters = [item for item in args.image_filters.split(',')]
    print('image_filters', image_filters)

    feature_preprocessing_method = args.feature_preprocessing_method
    feature_selection_methods_str = args.feature_selection_methods
    feature_selection_methods = {}
    # convert input string to dict
    for i in feature_selection_methods_str.split('/'):
        i_splited = i.split(',')
        assert len(i_splited) == 1 or len(i_splited) < 2
        if len(i_splited) >2:
            step, kw_name, kw_value = i.split(',')
            feature_selection_methods[int(step)] = {kw_name: kw_value}
        else:
            step = int(i_splited[0])
            feature_selection_methods[int(step)] = {}
    print(feature_selection_methods)

    print('feature_selection_methods', feature_selection_methods)
    print(feature_selection_methods.keys())
    classifier = [int(item) for item in args.classifier.split(',')]
    print('classifier', classifier)
    random_seed = args.random_seed

    # testing
    # train_dir = 'G:\GE/2018-2019_NPC_ROI\dataset/'
    # patient_info = 'G:\GE/2018-2019_NPC_ROI/info.xlsx'
    # save_dir = 'G:\GE/2018-2019_NPC_ROI/TestV2/RFSelectorFastmode/'
    # modility_for_extraction = ['T1WI-AX', 'T2WI-AX', 'DWI-AX']
    # radiomic_feature_types = ['first_order', 'shape3d', 'glcm', 'glrlm', 'glszm', 'ngtdm', 'gldm']
    # image_filters = ['original', 'log_1', 'log_3', 'log_5']
    # test_size = 0.3
    # # label_tag = 1
    # feature_preprocessing_method = 0
    # feature_selection_methods = {1: {"p_feature": 0.3}, 4: {"p_feature": 0.25}, 8: {}}
    # classifier = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # random_seed = 42
    #

    trainer = Trainer(data_dir, patient_info, test_size, save_dir, modality_for_extraction,
                      radiomic_feature_types, image_filters, feature_preprocessing_method, feature_selection_methods,
                      classifier, random_seed)
    trainer.run_process()
    # Todo:add load processed feature selection process
    # Todo: make lasso determinstic
    #


if __name__ == '__main__':
    main()
    # cmd command line
