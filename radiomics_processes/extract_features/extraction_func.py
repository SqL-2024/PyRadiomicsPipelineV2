from radiomics import firstorder
import six
import SimpleITK as sitk
import numpy as np
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape, shape2D, ngtdm, gldm
from copy import deepcopy
from scipy.ndimage.morphology import binary_dilation
import logging

# set level for all classes
# logger = logging.getLogger("radiomics")
# logger.setLevel(logging.ERROR)
# # ... or set level for specific class
# logger = logging.getLogger("radiomics.glcm")
# logger.setLevel(logging.ERROR)


def get_first_order_feature(image, mask, settings):
    firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)
    firstOrderFeatures.enableAllFeatures()
    result = firstOrderFeatures.execute()
    features_all = []
    feature_names = []
    for (key, val) in six.iteritems(result):
        features_all.append(val.tolist())
        feature_names.append(key)

    image_array = sitk.GetArrayFromImage(image).astype('float')
    mask_array = sitk.GetArrayFromImage(mask)
    target_voxel = image_array[mask_array > 0].astype('float')
    percentile25 = np.nanpercentile(target_voxel, 25)
    percentile75 = np.nanpercentile(target_voxel, 75)
    features_all.extend([percentile25, percentile75])
    feature_names.extend(['25Percentile', '75Percentile'])
    return features_all, feature_names


def get_shape3d_feature(image, mask, settings):
    shapeFeatures = shape.RadiomicsShape(image, mask, **settings)
    shapeFeatures.enableAllFeatures()
    result = shapeFeatures.execute()
    features_all = []
    feature_names = []
    for (key, val) in six.iteritems(result):
        features_all.append(val.tolist())
        feature_names.append(key)
    return features_all, feature_names


def get_shape2d_feature(image, mask, settings):
    shapeFeatures = shape2D.RadiomicsShape2D(image, mask, **settings)
    shapeFeatures.enableAllFeatures()
    result = shapeFeatures.execute()
    features_all = []
    feature_names = []
    for (key, val) in six.iteritems(result):
        features_all.append(val.tolist())
        feature_names.append(key)
    return features_all, feature_names


def get_glcm_feature(image, mask, settings):
    glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
    glcmFeatures.enableAllFeatures()
    result = glcmFeatures.execute()
    features_all = []
    feature_names = []
    for (key, val) in six.iteritems(result):
        features_all.append(val.tolist())
        feature_names.append(key)
    return features_all, feature_names


def get_glrlm_feature(image, mask, settings):
    glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
    glrlmFeatures.enableAllFeatures()
    result = glrlmFeatures.execute()
    features_all = []
    feature_names = []
    for (key, val) in six.iteritems(result):
        features_all.append(val.tolist())
        feature_names.append(key)
    return features_all, feature_names


def get_glszm_feature(image, mask, settings):
    glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
    glszmFeatures.enableAllFeatures()
    result = glszmFeatures.execute()
    features_all = []
    feature_names = []
    for (key, val) in six.iteritems(result):
        features_all.append(val.tolist())
        feature_names.append(key)
    return features_all, feature_names


def get_ngtdm_feature(image, mask, settings):
    ngtdmFeatures = ngtdm.RadiomicsNGTDM(image, mask, **settings)
    ngtdmFeatures.enableAllFeatures()
    result = ngtdmFeatures.execute()
    features_all = []
    feature_names = []
    for (key, val) in six.iteritems(result):
        features_all.append(val.tolist())
        feature_names.append(key)
    return features_all, feature_names


def get_gldm_feature(image, mask, settings):
    gldmFeatures = gldm.RadiomicsGLDM(image, mask, **settings)
    gldmFeatures.enableAllFeatures()
    result = gldmFeatures.execute()
    features_all = []
    feature_names = []
    for (key, val) in six.iteritems(result):
        features_all.append(val.tolist())
        feature_names.append(key)
    return features_all, feature_names


def extract_single(image, mask, settings, feature_types):
    """
    :param image:
    :param mask:
    :param binwith:
    :param feature_types:
    :return:
    """
    FUNCS = {
        'first_order': get_first_order_feature,
        'shape3d': get_shape3d_feature,
        'shape2d': get_shape2d_feature,
        'glcm': get_glcm_feature,
        'glrlm': get_glrlm_feature,
        'glszm': get_glszm_feature,
        'ngtdm': get_ngtdm_feature,
        'gldm': get_gldm_feature,
    }

    feature_values = []
    feature_names = []
    for feature_type in feature_types:
        extractor = FUNCS[feature_type]
        feature_value, feature_name = extractor(image, mask, settings)
        feature_values.append(feature_value)
        feature_names.append(feature_name)
    feature_values = np.array(np.hstack(feature_values))
    feature_names = np.hstack(feature_names)
    return feature_values, feature_names


def extract_process(image, mask, binwidth, feature_types, image_filters, label_tag=1):
    """

    :param image:
    :param mask:
    :param binwidth:
    :param feature_types:
    :param image_filters:
    :param label_tag:
    :return:
    """
    settings = {'binWidth': binwidth,
                'interpolator': sitk.sitkBSpline,
                'resampledPixelSpacing': None,
                'label': label_tag}
    bb, correctedMask = imageoperations.checkMask(image, mask, correctMask=True)
    if correctedMask is not None:
        mask = correctedMask
    mask_arr = sitk.GetArrayFromImage(mask)

    # if bb[-1] - bb[-2] <= 3:
    #     mask_arr_dilated = binary_dilation(mask_arr).astype(mask_arr.dtype)
    #     mask_new = sitk.GetImageFromArray(mask_arr_dilated)#Todo
    #     mask_new.CopyInformation(mask)
    #     mask = mask_new
    #     bb, correctedMask = imageoperations.checkMask(image, mask, correctMask=True)

    image, mask = imageoperations.cropToTumorMask(image, mask, bb)
    image = imageoperations.normalizeImage(image)
    feature_shape = [f for f in feature_types if f.startswith('shape')]
    feature_types_no_shape = deepcopy(feature_types)
    for i in range(len(feature_shape)):
        feature_types_no_shape.remove(feature_shape[i])
    print()
    # print('feature without shape is:', feature_types_no_shape)

    feature_all = []
    feature_name_all = []
    for im_filter in image_filters:
        # print(im_filter)
        if im_filter == 'original':
            for image_filtered, ty, kw in imageoperations.getOriginalImage(image, mask):
                feature_value, feature_name = extract_single(image_filtered, mask, settings, feature_types)
                # print(feature_value.shape)
                feature_name_new = [sub_feature_name + '_' + ty for sub_feature_name in feature_name]

        elif im_filter.startswith('log_'):
            # Check if size of image is > 4 in all 3D directions (otherwise, LoG filter will fail)
            # https://github.com/AIM-Harvard/pyradiomics/blob/master/radiomics/imageoperations.py
            # mask_arr = sitk.GetArrayFromImage(mask)
            # z_slice =
            # print(image.GetSize())
            sigma = [float(im_filter.split('_')[-1])]
            for image_filtered, ty, kw in imageoperations.getLoGImage(image, mask, sigma=sigma):
                feature_value, feature_name = extract_single(image_filtered, mask, settings, feature_types_no_shape)
                # print(feature_value.shape)

                feature_name_new = [sub_feature_name + '_' + im_filter for sub_feature_name in feature_name]


        elif im_filter == 'wavelet':
            feature_value = []
            feature_name_new = []
            for image_filtered, ty, kw in imageoperations.getWaveletImage(image, mask):
                # ty is wavelet-LLH wavelet-LHL wavelet-LHH wavelet-HLL wavelet-HLH wavelet-HHL wavelet-HHH wavelet-LLL
                # so concat first
                feature_value_sub, feature_name_sub = extract_single(image_filtered, mask, settings,
                                                                     feature_types_no_shape)
                feature_name = [sub_feature_name + '_' + ty for sub_feature_name in feature_name_sub]
                feature_value.append(feature_value_sub)
                feature_name_new.append(feature_name)
            feature_value = np.hstack(feature_value)
            feature_name_new = np.hstack(feature_name_new)

        elif im_filter == 'square':
            for image_filtered, ty, kw in imageoperations.getSquareImage(image, mask):
                feature_value, feature_name = extract_single(image_filtered, mask, settings, feature_types_no_shape)
                feature_name_new = [sub_feature_name + '_' + im_filter for sub_feature_name in feature_name]

        elif im_filter == 'square_root':
            for image_filtered, ty, kw in imageoperations.getSquareRootImage(image, mask):
                feature_value, feature_name = extract_single(image_filtered, mask, settings, feature_types_no_shape)
                feature_name_new = [sub_feature_name + '_' + im_filter for sub_feature_name in feature_name]

        elif im_filter == 'logarithm':
            for image_filtered, ty, kw in imageoperations.getLogarithmImage(image, mask):
                feature_value, feature_name = extract_single(image_filtered, mask, settings, feature_types_no_shape)
                feature_name_new = [sub_feature_name + '_' + im_filter for sub_feature_name in feature_name]
        elif im_filter == 'exponential':
            for image_filtered, ty, kw in imageoperations.getExponentialImage(image, mask):
                feature_value, feature_name = extract_single(image_filtered, mask, settings, feature_types_no_shape)
                feature_name_new = [sub_feature_name + '_' + im_filter for sub_feature_name in feature_name]
        elif im_filter == 'gradient':
            for image_filtered, ty, kw in imageoperations.getGradientImage(image, mask):
                feature_value, feature_name = extract_single(image_filtered, mask, settings, feature_types_no_shape)
                feature_name_new = [sub_feature_name + '_' + im_filter for sub_feature_name in feature_name]
        elif im_filter == 'lbp-2D':
            for image_filtered, ty, kw in imageoperations.getLBP2DImage(image, mask):
                feature_value, feature_name = extract_single(image_filtered, mask, settings, feature_types_no_shape)
                feature_name_new = [sub_feature_name + '_' + im_filter for sub_feature_name in feature_name]
        elif im_filter == 'lbp-3D':
            feature_value = []
            feature_name_new = []
            for image_filtered, ty, kw in imageoperations.getLBP3DImage(image, mask):
                # ty is  lbp-3D-m1 lbp-3D-m2 lbp-3D-k
                # concat first
                feature_value_sub, feature_name_sub = extract_single(image_filtered, mask, settings,
                                                                     feature_types_no_shape)
                feature_name = [sub_feature_name + '_' + ty for sub_feature_name in feature_name_sub]
                feature_value.append(feature_value_sub)
                feature_name_new.append(feature_name)
            feature_value = np.hstack(feature_value)
            feature_name_new = np.hstack(feature_name_new)
        feature_all.append(feature_value)
        # print(feature_value.shape)
        feature_name_all.append(feature_name_new)
    feature_all = np.array(np.hstack(feature_all))
    feature_name_all = np.hstack(feature_name_all)

    return feature_all, feature_name_all


if __name__ == '__main__':
    im = sitk.ReadImage('G:\GE/2018-2019_NPC_ROI\dataset/1075107\T1WI-AX/image.nii.gz')
    mask = sitk.ReadImage('G:\GE/2018-2019_NPC_ROI\dataset/1075107\T1WI-AX/mask.nii.gz')
    feature_types = ['first_order', 'shape3d', 'glcm', 'glrlm', 'glszm', 'ngtdm', 'gldm']
    image_filters = ['original', 'log_1', 'log_3', 'log_5', 'wavelet', 'square', 'gradient', 'lbp-3D']
    feature, name = extract_process(im, mask, 25, feature_types, image_filters)
    print(feature.shape)
