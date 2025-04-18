import pickle

import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
from radiomics_processes.extract_features.extraction_func import extract_process


class RadiomicFeatureExtractor(object):
    def __init__(self, image_mask_dir, patient_info, save_dir, dicom_modility, radiomic_feature_types, image_filters,
                 label_tag=1):
        """

        :param image_mask_dir:
        :param patient_info:
        :param save_dir:
        :param dicom_modility:
        :param label_tag:
        """
        self.image_mask_dir = image_mask_dir
        self.patient_info = patient_info
        self.save_dir = save_dir
        self.dicom_modility = dicom_modility
        self.label_tag = label_tag
        self.radiomic_feature_types = radiomic_feature_types
        self.image_filters = image_filters

    def get_binwidth(self, case_ids, modility):
        """
        automaticlly set binwidth,random sample 10 cases and get mean min and mean max,
        :param case_ids:
        :param modility:
        :return:
        """
        case_sampled = np.random.choice(case_ids, 10, replace=False)
        mins = []
        maxs = []
        for case in case_sampled:
            if not isinstance(case, str):
                case = str(case)
            image = sitk.ReadImage(os.path.join(self.image_mask_dir, case, modility, 'image.nii.gz'))
            mask = sitk.ReadImage(os.path.join(self.image_mask_dir, case, modility, 'mask.nii.gz'))
            imarr = sitk.GetArrayFromImage(image)
            maskarr = sitk.GetArrayFromImage(mask)
            taget_voxels = imarr[maskarr == self.label_tag]
            mins.append(taget_voxels.min())
            maxs.append(taget_voxels.max())
        mean_min = np.mean(mins)
        mean_max = np.max(maxs)
        return (mean_max - mean_min) / 10

    def run_extraction_case(self, case_id, modility):
        """
        :param case_id:
        :param modility:
        :return:
        """
        image = sitk.ReadImage(os.path.join(self.image_mask_dir, case_id, modility, 'image.nii.gz'))
        mask = sitk.ReadImage(os.path.join(self.image_mask_dir, case_id, modility, 'mask.nii.gz'))
        feature_value, feature_name = extract_process(image, mask, self.binwidth_dict[modility],
                                                      self.radiomic_feature_types,
                                                      self.image_filters, label_tag=self.label_tag)
        feature_name_new = [sub_feature_name + '_' + modility for sub_feature_name in feature_name]
        return feature_value, feature_name_new

    def run_extraction(self):
        case_info = pd.read_excel(self.patient_info)
        case_ids = case_info['case_id'].values
        labels = case_info['label'].values
        all_features = []

        self.binwidth_dict = {}
        for modility in self.dicom_modility:
            self.binwidth_dict[modility] = self.get_binwidth(case_ids, modility)

        for case in case_ids:
            # print(case)
            if not isinstance(case, str):
                case = str(case)
            case_feature = []
            case_feature_name = []

            for modility in self.dicom_modility:
                feature_value, feature_name = self.run_extraction_case(case, modility)
                case_feature.append(feature_value)
                case_feature_name.append(feature_name)
            all_features.append(np.hstack(case_feature))

            all_features_name = np.hstack(case_feature_name)
        all_case_features = np.stack(all_features)
        feature_data = pd.DataFrame(all_case_features, columns=all_features_name)
        feature_data.insert(0, 'case_ids', case_ids)
        feature_data.insert(1, 'label', labels)
        feature_data.to_excel(os.path.join(self.save_dir, 'radiomic_features.xlsx'), index=False)
        print('the number of feature extracted is:', feature_data.shape[1] - 2)
        return feature_data


if __name__ == '__main__':

