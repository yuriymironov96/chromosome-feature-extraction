import keras.api.utils as image
from keras.api.applications.vgg16 import VGG16
from keras.api.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import shutil
import glob
import os.path


image.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights='imagenet', include_top=False)

ROOT_INPUT_DIR = os.path.join(os.getcwd(), 'input_data')
ROOT_OUTPUT_DIR = os.path.join(os.getcwd(), 'output_data')
imdir = os.path.join(ROOT_INPUT_DIR, "clusterized_chromosomes")
targetdir = os.path.join(ROOT_OUTPUT_DIR, "clusterized_chromosomes_result")
number_clusters = 3

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.jpg'))
filelist = [file for file in filelist if not os.path.split(file)[-1].startswith("_")]
filelist.sort()

featurelist = []
filenames = []

for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(100, 100))
    # img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())
    filenames.append(imagepath.split('/')[-1])

# Clustering
kmeans = KMeans(n_clusters=number_clusters, init='k-means++', n_init=10, random_state=0).fit(np.array(featurelist))

# Copy images renamed by cluster
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass

from sklearn.metrics import rand_score, adjusted_rand_score
labels_true = []
labels_pred = []
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
