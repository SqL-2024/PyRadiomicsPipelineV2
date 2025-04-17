# check if there are missing of unmatched cases
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd


def verify_dataset_intergrity(image_mask_dir, patient_info, modality):
    """
    check if the data is intergrity,like missing mask

    :param folder:
    :return:
    """
    case_info = pd.read_excel(patient_info)
    case_ids = case_info['case_id'].values
    # nii_files_in_images = [name for name in os.listdir(os.path.join(folder, 'images')) if name.endswith('.nii.gz')]
    # nii_files_in_masks = [name for name in os.listdir(os.path.join(folder, 'masks')) if name.endswith('.nii.gz')]
    # take csv file as reference

    print("Verifying data set")
    for m in modality:
        for case in case_ids:
            # print("checking case", case)
            if not isinstance(case, str):
                case = str(case)
            expected_label_file = os.path.join(image_mask_dir, case, m, 'image.nii.gz')
            expected_image_files = os.path.join(image_mask_dir, case, m, 'mask.nii.gz')
            assert os.path.isfile(expected_label_file), "could not find label file for case %s. Expected file: \n%s" % (
                case, expected_label_file)
            assert os.path.isfile(
                expected_image_files), "could not find image file for case %s. Expected file: \n%s" % (
                case, expected_label_file)

    print("Dataset OK")
