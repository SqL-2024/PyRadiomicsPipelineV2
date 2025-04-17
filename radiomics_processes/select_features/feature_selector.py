import os
import pandas as pd
import numpy as np
import pickle
from radiomics_processes.select_features.selection_func import *
from radiomics_processes.metrics_reports.feature_visualization import plot_featur_heatmap


class FeatureSelector(object):
    def __init__(self, feature_selection_methods, save_dir):
        self.feature_selection_methods = feature_selection_methods
        self.save_dir = save_dir
        self.feature_selection_log_before = None
        self.feature_selection_method_all = {
            1: 'Chi2Test',
            2: 'ANOVA F-value',
            3: 'Mutual information',
            4: 'mRMR(minimum redundancy maximum relevance)',
            5: 'Relief',
            6: 'RFE(recursive feature elimination)',
            7: 'Lasso',  # two graph of loss and coefficient,one excel
            8: 'RandomForestImportance',  # graph of feature importance,one excel
            9: 'Sequential Feature Selection',
            10: 'Boruta'
        }
        self.FUNCS = {
            'Chi2Test': chi2_test_selector,
            'ANOVA F-value': anova_f_value_selector,
            'Mutual information': mutual_information_selector,
            'mRMR(minimum redundancy maximum relevance)': mRMR_selector,
            'Relief': refief_selector,
            'RFE(recursive feature elimination)': RFE_selector,
            'Lasso': lasso_selector,
            'RandomForestImportance': RF_selector,
            'Sequential Feature Selection': SFS_selector,
            'Boruta': Boruta_selector
        }

    def selector_base(self, selector_function, feature, label, kwargs):
        """
        general interface
        :param selector_function:
        :param feature:
        :param label:
        :param kwargs:
        :return:
        """
        support = selector_function(feature, label, self.save_dir, kwargs)
        return support

    def load_selected(self):
        file_name = os.path.join(self.save_dir, 'feature_selection_process.pkl')
        if os.path.exists(file_name):
            self.feature_selection_log_before = pickle.load(open(file_name, 'rb'))

    def run_selection(self, train_set, test_set):
        print('there are {} step in feature selection'.format(len(self.feature_selection_methods.keys())))

        feature_clean_train = train_set.drop(columns=['case_ids', 'label'])
        X_train = feature_clean_train.values
        y_train = train_set['label'].values
        feature_name = feature_clean_train.columns.values
        # if os.path.exists(os.path.join(self.save_dir, 'feature_selection_process.pkl'):
        #     feature_selection_log_before = pickle.load(open(os.path.join(self.save_dir, 'feature_selection_process.pkl'),'rb'))
        #     if
        # with open(os.path.join(self.save_dir, 'feature_selection_process.pkl'), 'wb') as f:
        #     pickle.dump(self.feature_selection_log, f)

        self.feature_selection_log = {}
        print('the all number of feature is {}'.format(X_train.shape[1]))
        step = 1
        for process in self.feature_selection_methods.keys():
            current_method = self.feature_selection_method_all[process]
            # if self.feature_selection_log_before is not None:
            #     if current_method==self.feature_selection_log_before[process]
            print('(step %d of %d) Processing with %s' % (step,
                                                          len(self.feature_selection_methods.keys()),
                                                          current_method))

            selector = self.FUNCS[current_method]  # corresponding selection function
            kwargs = self.feature_selection_methods[process]
            kwargs['feature_name'] = feature_name
            support = self.selector_base(selector, X_train, y_train, kwargs)
            X_train = X_train[:, support]
            feature_name = feature_name[support]
            self.feature_selection_log[process] = support
            step += 1
        with open(os.path.join(self.save_dir, 'feature_selection_process.pkl'), 'wb') as f:
            pickle.dump(self.feature_selection_log, f)
        new_train_set = self.get_feature_with_log(train_set, data_tag='train')
        new_test_set = self.get_feature_with_log(test_set, data_tag='test')
        return new_train_set, new_test_set

    def get_feature_with_log(self, feature_data, data_tag='train'):
        """

        :param feature_data:
        :return:
        """
        feature_clean_train = feature_data.drop(columns=['case_ids', 'label'], axis=1)  # remove non feature value
        labels = feature_data['label'].values  #
        case_ids = feature_data['case_ids'].values
        feature_value = feature_clean_train.values
        feature_name = feature_clean_train.columns.values
        selection_results = {}
        index_pd = []
        for process in self.feature_selection_log.keys():
            support = self.feature_selection_log[process]
            feature_value = feature_value[:, support]
            feature_name = feature_name[support]
            method = self.feature_selection_method_all[process]
            selection_results['after_' + method] = feature_name
            index_pd.append(method)
            if len(feature_name) <= 100:  # plot heatmap
                save_name = 'heatmap_of_{}_samples_after_{}.png'.format(data_tag, method)
                new_feature_df = pd.DataFrame(feature_value, columns=feature_name)
                plot_featur_heatmap(new_feature_df, self.save_dir, save_name)
        new_feature_data = pd.DataFrame(feature_value, columns=feature_name)
        new_feature_data.insert(0, 'case_ids', case_ids)
        new_feature_data.insert(1, 'label', labels)
        series = pd.DataFrame.from_dict(selection_results, orient='index')
        series.to_excel(os.path.join(self.save_dir, 'feature_selection_result.xlsx'), index=index_pd)
        new_feature_data.to_excel(os.path.join(self.save_dir, '{}_feature_after_selection.xlsx'.format(data_tag)),
                                  index=False)
        return new_feature_data
