from matplotlib import pyplot as plt

import numpy as np
from scipy.signal import argrelextrema
from scipy import ndimage

from get_ordered_coordinates import get_ordered_coordinates
from get_segments_from_relative_extrema import get_segments_from_relative_extrema
from utils import get_args


def get_segments_from_binary_image(binary_image, ordered_coordinates):
    plot = np.asarray([binary_image[coord[0], coord[1]] for coord in ordered_coordinates])
    # plot = ndimage.median_filter(plot, size=10)
    
    segments = []

    # is_new_segment = True
    # for idx, el in enumerate(plot[:-1]):
        
    #     segments.append(local_idx + extrema[idx][0])

    # print("binary_image", binary_image)
    # import pdb; pdb.set_trace()
    # print("ordered_coordinates", ordered_coordinates)
    # print("plot", plot)
    plt.figure(200)
    plt.plot(
        plot,
        range(len(ordered_coordinates)),
    )
    plt.xlabel("Chromosome colour")
    plt.ylabel("Chromosome length")
    plt.axis("scaled")
    # plt.show()

    return plot
