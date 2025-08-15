import numpy as np

class KITTICategory(object):

    CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    CLASS_MEAN_SIZE = {
        'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
        'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
        'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
    }

    CLASS_STD_SIZE = {
        'Car': np.array([0.43,0.1,0.14]),
        'Pedestrian': np.array([0.22, 0.15, 0.09]),
        'Cyclist': np.array([0.22, 0.15, 0.09]),
    }

    APP_POS = {
        'Car': np.array([-0.1, 0.6, 0.8]),
        'Pedestrian': np.array([0,0.8,0]),
        'Cyclist': np.array([-0.29, -0.7, -0.3]), 
    }

    NUM_SIZE_CLUSTER = len(CLASSES)

    MEAN_SIZE_ARRAY = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clusters
    for i in range(NUM_SIZE_CLUSTER):
        MEAN_SIZE_ARRAY[i, :] = CLASS_MEAN_SIZE[CLASSES[i]]

    STD_SIZE_ARRAY = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clusters
    for i in range(NUM_SIZE_CLUSTER):
        STD_SIZE_ARRAY[i, :] = CLASS_STD_SIZE[CLASSES[i]]
