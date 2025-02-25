import os
from matplotlib import pyplot as plt
from skimage.morphology import thin


import cv2

from get_ordered_coordinates import get_ordered_coordinates
from get_segments_from_relative_extrema import get_segments_from_relative_extrema
from get_segments_from_binary_image import get_segments_from_binary_image
from utils import get_args


def convert_to_binary(filepath: str):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    ret, otsu = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    return 255 - otsu


def prepare_image(filepath):
    original = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    gauss = cv2.GaussianBlur(original, (5, 5), 0)
    binary_img_orig = convert_to_binary(filepath)
    binary_img = cv2.resize(binary_img_orig, (0, 0), fx=0.5, fy=1)
    orig_thinned = thin(binary_img_orig).astype('uint8') * 255    
    thinned = thin(binary_img).astype('uint8') * 255    

    # https://stackoverflow.com/questions/72278844/python-image-preprocessing-thresholding-and-binarizing-low-contrast-images-fo
    # https://stackoverflow.com/questions/73971444/binarize-low-contrast-images
    # 37 looked good as well
    so_thresh = cv2.adaptiveThreshold(
        binary_img_orig,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        21,
        0
    )

    thinned_original_size = cv2.resize(thinned, (0, 0), fx=2, fy=1)

    return {
        "original": original,
        "gauss": gauss,
        "binary_img_orig": binary_img_orig,
        "binary_img": binary_img,
        "orig_thinned": orig_thinned,
        "thinned": thinned,
        "so_thresh": so_thresh,
        "thinned_original_size": thinned_original_size,
    }


def get_binary_skeleton_feature_from_image(filepath):
    primg = prepare_image(filepath)
    so_thresh = primg["so_thresh"]
    thinned_original_size = primg["thinned_original_size"]
    thinned = primg["thinned"]

    # original_size_ordered_coordinates = get_ordered_coordinates(thinned_original_size)
    original_size_ordered_coordinates = get_ordered_coordinates(thinned)
    original_size_ordered_coordinates[:, 1] *= 2

    return get_segments_from_binary_image(so_thresh, original_size_ordered_coordinates)


def get_skeleton_feature_from_image(filepath):
    primg = prepare_image(filepath)
    original = primg["original"]
    thinned_original_size = primg["thinned_original_size"]
    thinned = primg["thinned"]

    # original_size_ordered_coordinates = get_ordered_coordinates(thinned_original_size)
    original_size_ordered_coordinates = get_ordered_coordinates(thinned)
    original_size_ordered_coordinates[:, 1] *= 2

    return get_segments_from_relative_extrema(original, original_size_ordered_coordinates)["plot"]


def get_skeleton_relex_feature_from_image(filepath):
    primg = prepare_image(filepath)
    original = primg["original"]
    thinned_original_size = primg["thinned_original_size"]
    thinned = primg["thinned"]

    # original_size_ordered_coordinates = get_ordered_coordinates(thinned_original_size)
    original_size_ordered_coordinates = get_ordered_coordinates(thinned)
    original_size_ordered_coordinates[:, 1] *= 2

    return get_segments_from_relative_extrema(original, original_size_ordered_coordinates)["ideogram"]


def main(filepath):
    primg = prepare_image(filepath)
    original = primg["original"]
    binary_img_orig = primg["binary_img_orig"]
    binary_img = primg["binary_img"]
    thinned = primg["thinned"]
    gauss = primg["gauss"]
    orig_thinned = primg["orig_thinned"]

    # https://stackoverflow.com/questions/72278844/python-image-preprocessing-thresholding-and-binarizing-low-contrast-images-fo
    # https://stackoverflow.com/questions/73971444/binarize-low-contrast-images
    # 37 looked good as well
    so_thresh = cv2.adaptiveThreshold(
        original,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        21,
        0
    )

    ordered_coordinates = get_ordered_coordinates(orig_thinned)

    get_segments_from_relative_extrema(original, ordered_coordinates)

    thinned_original_size = cv2.resize(thinned, (0, 0), fx=2, fy=1)

    original_size_ordered_coordinates = get_ordered_coordinates(thinned)
    original_size_ordered_coordinates[:, 1] *= 2

    skeleton_on_original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB) & (255 - cv2.cvtColor(orig_thinned, cv2.COLOR_GRAY2RGB))
    

    # get_segments_from_binary_image(so_thresh, original_size_ordered_coordinates)

    images = [
        ("original", original),
        # ("binary", binary_img),
        # ("gauss", gauss),
        # ("adaptive mean", adap_mean),
        # ("adaptive gauss", adap_gauss),
        # ("so_thresh", so_thresh),
        # *thresholded_images,
        # ("otsu", otsu),
        # ("skeletonize", skeleton),
        # ("skeletonize (Lee 94)", skeleton_lee),
        # ("compressed_skeleton", thinned),
        # ("decompressed_skeleton", thinned_original_size),
        # ("plot", plot),
        # (f"thin {iter_count}", thinned_partial),
        # ("thin original size", thinned_original_size),
        ("skeleton+threshold", skeleton_on_original),
        # ("mapped skeleton on original img", skeleton_on_original),
    ]

    fig, axes = plt.subplots(1, len(images), figsize=(
        8, 4), sharex=False, sharey=True)
    ax = axes.ravel()

    for i, (title, img) in enumerate(images):
        ax[i].imshow(img, cmap=plt.cm.gray)
        ax[i].set_title(title)
        ax[i].axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = get_args()
    filepath = args["filepath"]
    is_dir = args["is_dir"]
    if is_dir:
        for file in os.listdir(filepath):
            if file.endswith(".png"):
                try:
                    main(os.path.join(filepath, file))
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
    else:
        main(filepath)
