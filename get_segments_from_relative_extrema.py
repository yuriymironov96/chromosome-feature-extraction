from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy import ndimage


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_distinct_extrema(array):
    groups = [[array[0]]]
    for el in array[1:]:
        if groups[-1][-1] + 1 == el:
            groups[-1].append(el)
        else:
            groups.append([el])
    return np.asarray([group[int(len(group) / 2)] for group in groups])


def get_segments_from_relative_extrema(img, ordered_coordinates):
    # print(ordered_coordinates)

    # [img[coord[0], coord[1]] for coord in ordered_coordinates]
    # plot = np.asarray([img[coord[0], coord[1]] for coord in ordered_coordinates[:-1]])
    plot = np.asarray([img[coord[0], coord[1]]
                      for coord in ordered_coordinates])
    plot = ndimage.gaussian_filter(plot, sigma=20)

    maxima = argrelextrema(plot, np.greater_equal, order=10)[0]
    maxima = [(value, True) for value in get_distinct_extrema(maxima)]
    # maxima = maxima[np.insert((np.diff(maxima) - 1).astype(np.bool), 0, True)]

    # print("maxima", maxima)

    minima = argrelextrema(plot, np.less_equal, order=10)[0]
    minima = [(value, False) for value in get_distinct_extrema(minima)]
    # print("minima", minima)
    extrema = maxima + minima
    extrema.sort()
    segments = []

    for idx, el in enumerate(extrema[:-1]):
        point_a = plot[extrema[idx][0]]
        point_b = plot[extrema[idx + 1][0]]
        threshold = (int(point_a) + int(point_b)) / 2

        min_value = min(point_a, point_b)
        if point_a > point_b:
            threshold = threshold - (threshold - min_value) * 0.2
        else:
            threshold = threshold - (threshold - min_value) * 0.2
        plot_segment = plot[extrema[idx][0]: extrema[idx + 1][0]]
        if len(plot_segment) == 0:
            continue
        realistic_threshold, local_idx = find_nearest(plot_segment, threshold)
        
        # if point_a > point_b:
        #     local_idx += int((local_idx - extrema[idx][0]) * 0.1)
        # else:
        #     local_idx -= int((extrema[idx + 1][0] - local_idx) * 0.1)
        # value = median_high(plot_segment)
        # target_idx = np.where(plot_segment == realistic_threshold)[0] + idx
        segments.append(local_idx + extrema[idx][0])

    # plt.plot(
    #     range(len(ordered_coordinates)),
    #     ndimage.gaussian_filter(plot, sigma=20),
    # )

    # argrelextrema(ndimage.gaussian_filter(plot, sigma=20), np.greater_equal)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    # https://stackoverflow.com/questions/56811444/finding-local-extreme-not-working-as-expected-in-scipy
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

    # import pdb; pdb.set_trace()
    # plt.plot(
    #     range(len(ordered_coordinates)),
    #     plot,
    # )
    # plt.xlabel("Chromosome length")
    # plt.ylabel("Chromosome colour")

    # for segment in segments:
    #     plt.plot([segment, segment], [50, 250], color="black")
    # for extrema_value in extrema:
    #     idx, _ = extrema_value
    #     plt.scatter(idx, plot[idx], marker="o", color="black")
    plt.plot(
        plot,
        range(len(ordered_coordinates)),
    )
    plt.xlabel("Chromosome colour")
    plt.ylabel("Chromosome length")
    plt.axis("scaled")

    abs_minima = 255
    abs_maxima = 0

    for extrema_value in extrema:
        idx, _ = extrema_value
        abs_maxima = max(abs_maxima, plot[idx])
        abs_minima = min(abs_minima, plot[idx])
        # plt.scatter(plot[idx], idx, marker="o", color="black")

    # for segment in segments:
    #     plt.plot([abs_minima, abs_maxima], [segment, segment], color="black")
        
    # plt.figure(300)
    # plt.show()
    plt.savefig("plot.png")
    # plt.imread()

    ideogram_rep = np.empty((len(plot), 15))
    ideogram = np.zeros(len(plot))
    

    visual_segments_data = [(f[0], f[1][1])
                            for f in zip(segments, extrema[:-1])]
    prev_idx = 0
    for seg in visual_segments_data:
        idx, is_white = seg
        color = 255 if is_white else 0
        for r in range(prev_idx, idx):
            ideogram_rep[r, :] = color
            ideogram[r] = color
        prev_idx = idx + 1

    # color = 255 if extrema[:-1][1] else 0
    # for r in range(prev_idx, len(plot)):
    #     ideogram_rep[r, :] = color

    return {
        "ideogram": ideogram,
        "plot": plot,
    }
