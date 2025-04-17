##################################################################
# The reference code for the work published in:
#  Wang X, **Liu S**, Yan Z, Yin F, Feng J, Liu H, Liu Y, Li Y.  
#   *Radiomics nomograms based on multi-sequence MRI for identifying cognitive impairment and predicting cognitive progression in relapsing-remitting multiple sclerosis.*  
#   **Academic Radiology.** 2025;32(1):411–424.  
#   [doi:10.1016/j.acra.2024.08.026](https://doi.org/10.1016/j.acra.2024.08.026) | PMID: [39198138](https://pubmed.ncbi.nlm.nih.gov/39198138/)
# Radiomics Feature Extraction from MRI images
# This script extracts radiomics features from MRI images using SimpleITK and pyradiomics.
# It includes preprocessing steps such as resampling, windowing, and feature extraction.
# The extracted features are saved in a specified directory.
# The script also handles the reading of image and mask files, and applies various filters to the images.
# The extracted features are stored in a dictionary and saved as pickle files.
##################################################################
import pickle
import os
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
from radiomics_processes.extract_features.extraction_func import extract_process
import pandas as pd
from batchgenerators.augmentations.utils import resize_segmentation
import radiomics
import warnings

warnings.simplefilter('ignore', DeprecationWarning)
logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)


def get_new_mask(original_nii):
    original_arr = sitk.GetArrayFromImage(original_nii)
    new_arr = (original_arr > 1).astype(original_arr.dtype)
    new_nii = sitk.GetImageFromArray(new_arr)
    new_nii.CopyInformation(original_nii)
    return new_nii


def resize_volume(volume, target_spacing, order=3, cval=0, is_seg=False):
    # # # with skimage.transforms.resize
    # # # order:0: Nearest-neighbor   1: Bi-linear (default)  2: Bi-quadratic   3: Bi-cubic  4: Bi-quartic   5: Bi-quintic
    # # # Constant padding value if image is not perfectly divisible by the integer factors.padding with 0
    data = sitk.GetArrayFromImage(volume)
    shape = data.shape
    original_spacing = [volume.GetSpacing()[2], volume.GetSpacing()[0], volume.GetSpacing()[1]]
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(
        int)
    # print('original spacing and shape is :', original_spacing, shape)
    # print('new spacing and shape is :', target_spacing, new_shape)
    if is_seg:
        kwargs = {'mode': 'edge', 'clip': True, 'anti_aliasing': True}
    else:
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    data = data.astype(float)
    new_shape = np.array(new_shape)
    resampled_data = resize(data, new_shape, order, cval=cval, **kwargs)
    if is_seg:
        resampled_data[resampled_data >= 0.5] = 1

    resampled_data = resampled_data.astype(dtype_data)
    resampled_nii = sitk.GetImageFromArray(resampled_data)
    resampled_nii.SetOrigin(volume.GetOrigin())
    resampled_nii.SetSpacing(tuple([target_spacing[1], target_spacing[2], target_spacing[0]]))
    resampled_nii.SetDirection(volume.GetDirection())
    return resampled_nii


if __name__ == '__main__':

    radiomic_feature_types = ['first_order', 'shape3d', 'glcm', 'glrlm', 'glszm', 'ngtdm', 'gldm']
    # feature type
    image_filters = ['original', 'wavelet', 'square', 'gradient', 'lbp-2D', 'lbp-3D', 'logarithm', 'exponential',
                     'square_root']
    # image filter to generate new images
    main_dir = '.\image_dirs'
    # clip to np.percentile(ima,00.5),np.percentile(ima,99.5)
    with open('./file_name_info.pkl', 'rb') as f:
        file_infos = pickle.load(f)
    modality_list = [ 'FLAIR',  'T1WI', 'DIR']
    target_spacing = [5.0, 0.75, 0.75]  # to resample image and mask
    for modality in modality_list:  # for each modality
        print('current modality is {}'.format(modality))
        modality_feature = []
       
        case_ids = []
        for key in list(file_infos.keys()):  # for each case id
            print('current case is {}'.format(key))
            case_ids.append(key)
            case_info = file_infos[key]
            im = sitk.ReadImage(case_info['{}_im'.format(modality)])  # read the modality image of this case
            im_arr = sitk.GetArrayFromImage(im)
            im_arr_new = np.clip(im_arr, np.percentile(im_arr, 0.5), np.percentile(im_arr, 99.5))
            # clip the image to (0.5,99.5)
            im_new = sitk.GetImageFromArray(im_arr_new)
            im_new.CopyInformation(im)
            im_resampled = resize_volume(im_new, target_spacing)  # resample image

            mask = sitk.ReadImage(case_info['{}_seg'.format(modality)])
            mask_resampled = resize_volume(mask, target_spacing, order=0, is_seg=True)  # resample mask

            # extract radiomics features
            feature_value, feature_name = extract_process(im_resampled, mask_resampled, 50,
                                                          radiomic_feature_types,
                                                          image_filters)
            print('orginal_mask features')
            if feature_value.shape[0] != 1724:
                print(print('case {} original_mask shape is {}'.format(key, feature_value.shape[0])))
                break
            modality_feature.append(feature_value)

          

        all_features_name = feature_name
        all_modality_feature = np.stack(modality_feature)
        feature_data = pd.DataFrame(all_modality_feature, columns=all_features_name)
        feature_data.insert(0, 'case_ids', case_ids)

        feature_data.to_csv(
            os.path.join('./radiomics_features', '{}_original.csv'.format(modality)), index=False)
        print('the number of feature extracted is:', feature_data.shape[1] - 1)

        