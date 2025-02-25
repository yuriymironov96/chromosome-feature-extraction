import numpy as np


def is_adjacent(coordinate_a, coordinate_b) -> bool:
    if (
        abs(coordinate_a[0] - coordinate_b[0]) <= 1
        and abs(coordinate_a[1] - coordinate_b[1]) <= 1
    ):
        # print(f"these nodes are adjacent {coordinate_a} {coordinate_b}")
        return True
    return False


def is_leaf(target_idx, coordinates) -> bool:
    """Checks if coordinates by target_idx are the ending point of curve."""

    found_one_adjacent = False
    for idx in range(coordinates.shape[0]):

        # skip iteration if we consider the same indices
        if idx == target_idx:
            continue

        if is_adjacent(coordinates[target_idx], coordinates[idx]):
            # we allow only one entering inside this if branch. If we get here twice - this
            # is not a leaf.
            if found_one_adjacent:
                return False
            found_one_adjacent = True
    return True


def get_ordered_coordinates(thinned_image):
    coordinates = np.transpose((thinned_image > 0).nonzero())
    ordered_coordinates = []
    count = 0

    # import pdb;pdb.set_trace()

    # find first node and assign it to first array element
    for idx in range(coordinates.shape[0]):
        if is_leaf(idx, coordinates):
            ordered_coordinates.append(coordinates[idx])
            count += 1
            break

    # import pdb; pdb.set_trace()
    for iteration in range(1, coordinates.shape[0]):
        coord = ordered_coordinates[-1]
        for idx in range(coordinates.shape[0]):
            if is_adjacent(coord, coordinates[idx]):

                # skip already assigned point
                if iteration > 1 and np.array_equal(
                    ordered_coordinates[-2], coordinates[idx]
                ):
                    continue
                if np.array_equal(ordered_coordinates[-1], coordinates[idx]):
                    continue
                # print(f"found adjacent nodes: {coord} {coordinates[idx]}")
                # assign adjacent point
                ordered_coordinates.append(coordinates[idx])
                count += 1
                break
    
    # print("len", ordered_coordinates.shape[0])
    return np.array(ordered_coordinates)
