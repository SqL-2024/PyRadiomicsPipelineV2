##################################################################
# The reference code for the work published in:
# Du J, He X, Fan R, Zhang Y, Liu H, Liu H, **Liu S**, Li S.  
#   *Artificial intelligence-assisted precise preoperative prediction of lateral cervical lymph nodes metastasis in papillary thyroid carcinoma via a clinical-CT radiomic combined model.*  
#   **International Journal of Surgery.** 2025;111(3):2453–2466.  
#   [doi:10.1097/JS9.0000000000002267](https://doi.org/10.1097/JS9.0000000000002267) | PMID: [39903541](https://pubmed.ncbi.nlm.nih.gov/39903541/)
#
# Radiomics Feature Extraction from CT images
# This script extracts radiomics features from CT images using SimpleITK and pyradiomics.
# It includes preprocessing steps such as resampling, windowing, and feature extraction.
# The extracted features are saved in a specified directory.
# The script also handles the reading of image and mask files, and applies various filters to the images.
# The extracted features are stored in a dictionary and saved as pickle files.
##################################################################
import glob
import pickle
import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
from radiomics_processes.extract_features.extraction_func import extract_process
# from xpinyin import Pinyin
from collections import OrderedDict
import json


def get_new_mask(original_nii):
    original_arr = sitk.GetArrayFromImage(original_nii)
    new_arr = (original_arr > 0).astype(original_arr.dtype)
    new_nii = sitk.GetImageFromArray(new_arr)
    new_nii.CopyInformation(original_nii)
    return new_nii


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


#
def change_window(original_nii, center=40, width=400):
    original_arr = sitk.GetArrayFromImage(original_nii)
    min = (2 * center - width) / 2.0 + 0.5  # get the mim value
    max = (2 * center + width) / 2.0 + 0.5  # get the max value
    dFactor = 255.0 / (max - min)
    img_temp = (original_arr - min) * dFactor
    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255.
    new_nii = sitk.GetImageFromArray(img_temp)
    new_nii.CopyInformation(original_nii)
    return new_nii


if __name__ == '__main__':

    main_dir = '../AllNiis/'
    save_dir = './radiomics_features'
    os.makedirs(save_dir, exist_ok=True)
    save_dir_a = os.path.join(save_dir, 'phase_a')
    os.makedirs(save_dir_a, exist_ok=True)
    save_dir_p = os.path.join(save_dir, 'phase_p')
    os.makedirs(save_dir_p, exist_ok=True)
    save_dir_v = os.path.join(save_dir, 'phase_v')
    os.makedirs(save_dir_v, exist_ok=True)


    radiomic_feature_types = ['first_order', 'shape3d', 'glcm', 'glrlm', 'glszm', 'ngtdm', 'gldm']
    image_filters = ['original', 'wavelet', 'square', 'gradient', 'lbp-2D', 'lbp-3D', 'exponential', 'logarithm',
                     'square_root']
    # target_spacing is the 50% percentile of the all spacings
    target_spacing = [0.48, 0.48, 1]

    # window processing
    window_level = 40
    window_width = 400
   
    i = 0
    with open('./Three_case_label.pkl', 'rb') as f:  # three
        case_infos = pickle.load(f)

    for case_id in case_ids:
        
        print('the {}th case is {}'.format(i, case_id))
        case_dir = os.path.join(main_dir, case_id)
        
        label_file = os.path.join(case_dir, 'label.json')
        if os.path.exists(label_file):  #
            with open(label_file) as f:
                label_info = json.load(f)
                label_tag = list(label_info['Models']['LabelDetailModel']['table'].keys())[0]
                gender = label_info['Models']['LabelDetailModel']['table'][label_tag]['性别']  # get label info
                age = label_info['Models']['LabelDetailModel']['table'][label_tag]['年龄']
                transfer = label_info['Models']['LabelDetailModel']['table'][label_tag]['转移']
        else:
            label_info = case_infos[case_id]
            gender = label_info['性别']
            age = label_info['年龄']
            transfer = label_info['转移']
        mask_dir = os.path.join(case_dir, 'Untitled.nii.gz')
        if os.path.exists(mask_dir):
            os.rename(mask_dir, os.path.join(case_dir, 'mask.nii.gz'))

        mask = sitk.ReadImage(os.path.join(case_dir, 'mask.nii.gz'))  # [0,2]
        print('mask size is:', mask.GetSize())
        new_mask = get_new_mask(mask)  # [0,1]
        resample_mask = resample_img(new_mask, target_spacing, is_label=True)

        if os.path.exists(os.path.join(case_dir, 'a.nii.gz')):  # if have a
            im = sitk.ReadImage(os.path.join(case_dir, 'a.nii.gz'))  # read original image
            print("a image size is:", im.GetSize())
            if im.GetSize() == mask.GetSize():  #
                # continue
                im.SetOrigin(mask.GetOrigin())
                im.SetDirection(mask.GetDirection())
                im.SetSpacing(mask.GetSpacing())
                featur_dict = OrderedDict()
                # resample to the target spacing
                resamples_im = resample_img(im, target_spacing, is_label=False)
                # window preprocessing or clip
                windowed_im = change_window(resamples_im, window_level, window_width)
                # clip
                resamples_im_arr = sitk.GetArrayFromImage(resamples_im)  #
                resamples_im_arr = np.clip(resamples_im_arr, np.percentile(resamples_im_arr, 0.5),
                                           np.percentile(resamples_im_arr, 99.5))
                new_im = sitk.GetImageFromArray(resamples_im_arr)
                new_im.CopyInformation(resamples_im)
                feature_value, feature_name = extract_process(new_im, resample_mask, 10,
                                                              radiomic_feature_types,
                                                              image_filters)

                if feature_value.shape[0] != 1724:
                    print('case {} phase A shape is {}'.format(case_id, feature_value.shape[0]))
                    break
                featur_dict['original'] = feature_value

                feature_value, feature_name = extract_process(windowed_im, resample_mask, 5,
                                                              radiomic_feature_types,
                                                              image_filters)
                featur_dict['windowed'] = feature_value
                featur_dict['name'] = feature_name
                featur_dict['label'] = transfer
                featur_dict['gender'] = gender
                featur_dict['age'] = age
                with open(os.path.join(save_dir_a, case_id + '.pkl'), 'wb') as f:
                    pickle.dump(featur_dict, f)
            else:
                print('a is not the same size')
        else:
            print('no a image, you should check it！')
        i += 1
