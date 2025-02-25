import cv2

import keras.api.utils as image
from keras.api.applications.vgg16 import VGG16
from keras.api.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import shutil
import glob
import os.path

# from src import get_segments_from_binary_image
from skeleton import get_binary_skeleton_feature_from_image

def resize(arr, new_length):
    arr = np.array(arr)
    n = len(arr)
    
    if new_length == n:
        return arr  # If the new length is the same as the original, just return the original array
    
    # Calculate the scaling factor
    scale = (n - 1) / (new_length - 1)
    
    # Create the indices for the new array by scaling and rounding them
    indices = np.linspace(0, n - 1, new_length)
    indices = np.round(indices * scale).astype(int)
    
    # Return the new array by selecting the elements based on the scaled indices
    return arr[indices]


image.LOAD_TRUNCATED_IMAGES = True
# model = VGG16(weights='imagenet', include_top=False)

ROOT_INPUT_DIR = os.path.join(os.getcwd(), 'input_data')
ROOT_OUTPUT_DIR = os.path.join(os.getcwd(), 'output_data')
imdir = os.path.join(ROOT_INPUT_DIR, "clusterized_chromosomes")
targetdir = os.path.join(ROOT_OUTPUT_DIR, "clusterized_chromosomes_binary_skeleton_result")
number_clusters = 3

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.jpg'))
filelist = [file for file in filelist if not os.path.split(file)[-1].startswith("_")]
filelist.sort()

featurelist = []
filenames = []

max_length = 0

for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    print("    Processing: %s" %imagepath, end="\r")
    # img = image.load_img(imagepath, target_size=(100, 100))
    # img = image.load_img(imagepath, target_size=(224, 224))
    # img_data = image.img_to_array(img)
    # features = np.array(model.predict(img_data))
    # features = np.array()
    # so_thresh = cv2.adaptiveThreshold(
    #     img,
    #     255,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,
    #     cv2.THRESH_BINARY,
    #     21,
    #     0
    # )
    # features = get_segments_from_binary_image(so_thresh, )
    feature = get_binary_skeleton_feature_from_image(imagepath)
    featurelist.append(feature)
    max_length = max(max_length, feature.shape[0])
    filenames.append(imagepath.split('/')[-1])

# proportionally resize all features to max_length
featurelist = [resize(feature, max_length) for feature in featurelist]

# Clustering
kmeans = KMeans(
    n_clusters=number_clusters,
    init='k-means++',
    n_init=10,
    random_state=0
).fit(np.array(featurelist))

from sklearn.metrics import rand_score, adjusted_rand_score
labels_true = []
labels_pred = []

# Copy images renamed by cluster
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    labels_pred.append(m)
    labels_true.append(int(filenames[i].split("_")[0]))
    shutil.copy(filelist[i], os.path.join(targetdir, str(m) + "__" + filenames[i]))

sklearn_rand_score = rand_score(labels_true, labels_pred)
sklearn_adjusted_rand_score = adjusted_rand_score(labels_true, labels_pred)
print("rand score: ", sklearn_rand_score)
print("adjusted rand score: ", sklearn_adjusted_rand_score)
print("results:", [{"cluster": str(m), "filename": filenames[i]}
      for i, m in enumerate(kmeans.labels_)])
