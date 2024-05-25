import random
import numpy as np
import matplotlib.pyplot as plt
import math

# the only parameters here that make sense are: 2 1; 3 2; 3 1
BEG_DIMENSIONS = 2  # >= 2
END_DIMENSIONS = 1  # 1 / 2; < beg_dimensions

NUM_OF_RED_PTS = 30  # > 0
NUM_OF_BLUE_PTS = 70  # > 0
NUM_OF_PURPLE_PTS = 50  # >= 0; set to 0 to only work with 2 data classes; set to >= 1 if end_dimensions == 2
OVERALL_NUM_OF_PTS = NUM_OF_RED_PTS + NUM_OF_BLUE_PTS + NUM_OF_PURPLE_PTS

RED_MIDPOINT_RANGE = [-5, 0]  # for simulation clarity; seeds for each data class to make them somewhat separated
BLUE_MIDPOINT_RANGE = [0, 5]
PURPLE_MIDPOINT_RANGE = [-6, -5]

RED_CLASS_SCALE = 1  # as above - approx sizes of point clusters in each class
BLUE_CLASS_SCALE = 2
PURPLE_CLASS_SCALE = 1.5

def create_dataset():
    """generates the coordinates of all points from all classes, as specified in the above settings"""

    def create_points(num_of_pts, midpoint_range, scale):
        # each of the lists in points[] represents one coordinate of all points in a given class
        # i.e. first list has the x coordinate of all points in this class; 2nd list has y coordinates; etc
        points = []
        for dimensions in range(BEG_DIMENSIONS):
            points.append([])

        # generating a point around which the class will be clustered
        # done for simulation clarity, so that the clusters aren't completely overlapping
        # so that good LDA projection is possible
        midpoint = [random.randint(midpoint_range[0], midpoint_range[1]) for i in range(BEG_DIMENSIONS)]

        # points are randomly scattered around the midpoint
        for pt_index in range(num_of_pts):
            for current_dim in range(BEG_DIMENSIONS):
                points[current_dim].append(np.random.normal(midpoint[current_dim], scale))
        return points

    red_points = create_points(NUM_OF_RED_PTS, RED_MIDPOINT_RANGE, RED_CLASS_SCALE)
    blue_points = create_points(NUM_OF_BLUE_PTS, BLUE_MIDPOINT_RANGE, BLUE_CLASS_SCALE)
    purple_points = create_points(NUM_OF_PURPLE_PTS, PURPLE_MIDPOINT_RANGE, PURPLE_CLASS_SCALE)

    return red_points, blue_points, purple_points


def compute_means(red_points, blue_points, purple_points):
    """computes the mean point of all clusters, and an overall midpoint of the whole dataset"""

    red_mean = []
    blue_mean = []
    purple_mean = []
    overall_mean = []

    for current_dim in range(BEG_DIMENSIONS):
        red_mean.append(np.mean(red_points[current_dim]))
        blue_mean.append(np.mean(blue_points[current_dim]))
        if NUM_OF_PURPLE_PTS == 0:
            purple_mean = None
            overall_mean.append((NUM_OF_RED_PTS * red_mean[current_dim] + NUM_OF_BLUE_PTS * blue_mean[
                current_dim]) / OVERALL_NUM_OF_PTS)  # weighed mean
        else:
            purple_mean.append(np.mean(purple_points[current_dim]))
            overall_mean.append((NUM_OF_RED_PTS * red_mean[current_dim] + NUM_OF_BLUE_PTS * blue_mean[
                current_dim] + NUM_OF_PURPLE_PTS * purple_mean[current_dim]) / OVERALL_NUM_OF_PTS)  # weighed mean

    return red_mean, blue_mean, purple_mean, overall_mean


def create_scatter_plot_2d(red_points, blue_points, purple_points):
    """creates a 2-dimensional scatter plot of all points in a dataset of 2-dimensional points"""

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.xlim(-12, 10)
    plt.ylim(-12, 10)
    ax.scatter(red_points[0], red_points[1], c="red")
    ax.scatter(blue_points[0], blue_points[1], c="blue")
    ax.scatter(purple_points[0], purple_points[1], c="purple")

    return fig, ax


def create_scatter_plot_3d(red_points, blue_points, purple_points):
    """creates a 3-dimensional scatter plot of all points in a dataset of 3-dimensional points"""

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')

    plt.xlim(-12, 10)
    plt.ylim(-12, 10)
    ax.set_zlim(-12, 10)

    ax.scatter(red_points[0], red_points[1], red_points[2], c="red")
    ax.scatter(blue_points[0], blue_points[1], blue_points[2], c="blue")
    ax.scatter(purple_points[0], purple_points[1], purple_points[2], c="purple")

    return fig, ax


def compute_within_class_scatter_matrix(red_points, red_mean, blue_points, blue_mean, purple_points, purple_mean):
    """computes the overall scatter matrix of all points within the dataset; returns a BEG_DIM x BEG_DIM matrix"""

    def compute_scatter_within_one_class(points, mean_point):
        """computes scatter of points within one class; returns a BEG_DIM x BEG_DIM matrix"""

        final_matrix = np.zeros((BEG_DIMENSIONS, BEG_DIMENSIONS))
        for point_id in range(len(points[0])):
            current_point = [[]]
            for current_dim in range(BEG_DIMENSIONS):
                current_point[0].append(points[current_dim][point_id])
            current_point = np.subtract(current_point, mean_point)
            final_matrix = np.add(final_matrix, (np.matmul(np.transpose(current_point), current_point)))

        return final_matrix

    red_within_scatter_matrix = compute_scatter_within_one_class(red_points, red_mean)
    blue_within_scatter_matrix = compute_scatter_within_one_class(blue_points, blue_mean)
    purple_within_scatter_matrix = compute_scatter_within_one_class(purple_points, purple_mean)
    overall_within_scatter_matrix = np.add(np.add(red_within_scatter_matrix, blue_within_scatter_matrix),
                                           purple_within_scatter_matrix)

    return overall_within_scatter_matrix


def compute_between_class_scatter_matrix(overall_mean, red_mean, blue_mean, purple_mean):
    def compute_scatter_between_one_class(class_mean, overall_mean, sample_size):
        """computes the distance between a given class midpoint and the overall dataset midpoint; returns a BEG_DIM x
        BEG_DIM matrix"""

        if sample_size == 0:
            return np.zeros((BEG_DIMENSIONS, BEG_DIMENSIONS))

        mean_diff = [np.subtract(class_mean, overall_mean)]
        final_matrix = np.matmul(np.transpose(mean_diff), mean_diff)
        final_matrix = np.multiply(final_matrix, sample_size)

        return final_matrix

    red_between_scatter_matrix = compute_scatter_between_one_class(red_mean, overall_mean, NUM_OF_RED_PTS)
    blue_between_scatter_matrix = compute_scatter_between_one_class(blue_mean, overall_mean, NUM_OF_BLUE_PTS)
    purple_between_scatter_matrix = compute_scatter_between_one_class(purple_mean, overall_mean, NUM_OF_PURPLE_PTS)
    overall_between_scatter_matrix = np.add(np.add(red_between_scatter_matrix, blue_between_scatter_matrix),
                                            purple_between_scatter_matrix)

    return overall_between_scatter_matrix


def compute_max_eigenvectors(within_matrix, between_matrix):
    """computes the eigenvectors and eigenvalues of the inverted within scatter matrix multiplied by the between
    scatter matrix; returns END_DIM eigenvectors corresponding to the END_DIM largest eigenvalues (or throws an error
    message when there is less non-zero eigenvalues than END_DIM and in that case returns the max possible amount of
    eigenvectors corresponding to these non-zero eigenvalues); these eigenvectors will then be the axes spanning the
    new subspace that the original datapoints will be projected onto"""

    within_inv = np.linalg.inv(within_matrix)
    final_matrix = np.matmul(within_inv, between_matrix)

    eig_vals, eig_vecs = np.linalg.eig(final_matrix)
    eig_vals = np.real(eig_vals)

    sorting_permutation_eigval = np.argsort(eig_vals)
    max_eigvecs = []
    for n in range(END_DIMENSIONS):
        if round(eig_vals[sorting_permutation_eigval[-n - 1]], 10) != 0:  # rounding necessary bc of flp inaccuracies
            max_eigvecs.append(eig_vecs[:, sorting_permutation_eigval[-n-1]])
        else:
            print(f"sorry, the maximum dimension of discriminant axes that create the subspace this set of points can "
                  f"be projected to is {n}")
            break
    return max_eigvecs


def proj(line_to_project_onto, vector):
    """projects a vector/point onto a given 1-dimensional subspace given a basis vector of such subspace; returns a
    list of coordinates of the new point"""

    scale_by = np.inner(vector, line_to_project_onto) / np.inner(line_to_project_onto, line_to_project_onto)
    return np.multiply(line_to_project_onto, scale_by)


def gram_schmidt_basis_orthogonalize(max_eigenvectors):
    """orthogonalizes a given set of vectors according to the Gram-Schmidt process; returns a set of orthogonal
    vectors"""

    orthogonal_basis = [0] * len(max_eigenvectors)
    for eigvec_id in range(len(max_eigenvectors)):
        temp_sum = 0
        for i in range(eigvec_id):
            temp_sum += proj(orthogonal_basis[i], max_eigenvectors[eigvec_id])
        orthogonal_basis[eigvec_id] = np.subtract(max_eigenvectors[eigvec_id], temp_sum)

    return orthogonal_basis


def normalise_basis(basis_vectors_list):
    """normalizes a given set of vectors by dividing every vector by its norm; returns a list of such vectors"""

    normalised_basis_vectors = [0] * len(basis_vectors_list)

    for vector_id in range(len(basis_vectors_list)):
        current_vector = basis_vectors_list[vector_id]
        normalised_basis_vectors[vector_id] = np.divide(current_vector, math.sqrt(np.real(np.inner(current_vector,
                                                                                                   current_vector))))

    return normalised_basis_vectors


def transform_pts(original_points, basis_vectors):
    """orthogonally projects the original datapoints onto a given subspace; returns the new coordinates of points
    (each coordinate in a separate list, as with the original datapoint coordinates)"""

    transformed_points = []
    for dim in range(BEG_DIMENSIONS):  # result has the same amount of dimensions as BEG_DIM, bc we never change basis
        transformed_points.append([])

    if len(original_points[0]) == 0:
        return transformed_points

    for point_id in range(NUM_OF_RED_PTS):
        current_point = [[]]
        for current_dim in range(BEG_DIMENSIONS):
            current_point[0].append(original_points[current_dim][point_id])

        if END_DIMENSIONS == 1:
            curr_pt_transformed = np.multiply(np.inner(current_point, basis_vectors[0]), basis_vectors[0])
        else:
            new_subspace_orthonormal_basis = normalise_basis(gram_schmidt_basis_orthogonalize(basis_vectors))
            curr_pt_transformed = [0] * BEG_DIMENSIONS
            for basis_vec in new_subspace_orthonormal_basis:
                curr_pt_transformed = np.add(curr_pt_transformed, proj(basis_vec, current_point))
        for current_dim in range(BEG_DIMENSIONS):
            transformed_points[current_dim].append(curr_pt_transformed[current_dim])

    return transformed_points


def update_plot_2d(fig, ax, orange_points, lightblue_points, lightpurple_points, vector):
    """updates the 2-dimensional scatter plot by adding the projected points and axis of projection, if we were
    projecting onto a 1-dimensional subspace; returns the plot specification"""

    ax.scatter(orange_points[0], orange_points[1], c="orange")
    ax.scatter(lightblue_points[0], lightblue_points[1], c="deepskyblue")
    ax.scatter(lightpurple_points[0], lightpurple_points[1], c="violet")

    if END_DIMENSIONS == 1:
        ax.axline((0, 0), (vector[0], vector[1]), color="black", linestyle="dashed")

    return fig, ax


def update_plot_3d(fig, ax, orange_points, lightblue_points, lightpurple_points, max_eigenvectors):
    """updates the 3-dimensional scatter plot by adding the projected points and the line/plane of projection
    (depending on whether we were projecting into 1-d or 2-d subspace; returns the plot specification"""

    ax.scatter(orange_points[0], orange_points[1], orange_points[2], c="orange")
    ax.scatter(lightblue_points[0], lightblue_points[1], lightblue_points[2], c="deepskyblue")
    ax.scatter(lightpurple_points[0], lightpurple_points[1], lightpurple_points[2], c="violet")

    if END_DIMENSIONS == 1:
        ax.plot((0, 100 * max_eigenvectors[0][0]), (0, 100 * max_eigenvectors[0][1]),
                (0, 100 * max_eigenvectors[0][2]), color="black", linestyle="dashed")
        ax.plot((0, -100 * max_eigenvectors[0][0]), (0, -100 * max_eigenvectors[0][1]),
                (0, -100 * max_eigenvectors[0][2]), color="black", linestyle="dashed")
    elif END_DIMENSIONS == 2:
        def plane_equation(x, y, max_eigenvectors):
            eig_vec_1 = max_eigenvectors[0]
            eig_vec_2 = max_eigenvectors[1]
            normal_vec_to_plane = np.cross(eig_vec_1, eig_vec_2)
            point_on_plane = eig_vec_1

            a = normal_vec_to_plane[0]
            b = normal_vec_to_plane[1]
            c = normal_vec_to_plane[2]
            x0 = point_on_plane[0]
            y0 = point_on_plane[1]
            z0 = point_on_plane[2]

            return (a * (x - x0) + b * (y - y0) - c * z0) / -c

        x_axis = [*range(-12, 13)]
        y_axis = [*range(-12, 13)]

        X, Y = np.meshgrid(x_axis, y_axis)
        Z = plane_equation(X, Y, max_eigenvectors)

        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, linewidth=0.5)

    return fig, ax


def main():
    red_points, blue_points, purple_points = create_dataset()

    red_mean, blue_mean, purple_mean, overall_mean = compute_means(red_points, blue_points, purple_points)

    if BEG_DIMENSIONS == 2:
        fig, ax = create_scatter_plot_2d(red_points, blue_points, purple_points)
    elif BEG_DIMENSIONS == 3:
        fig, ax = create_scatter_plot_3d(red_points, blue_points, purple_points)
    else:
        print(f"it currently isn't possible to create a {BEG_DIMENSIONS}-dimensional plot")

    overall_within_scatter_matrix = compute_within_class_scatter_matrix(red_points, red_mean,
                                                                        blue_points, blue_mean,
                                                                        purple_points, purple_mean)

    overall_between_scatter_matrix = compute_between_class_scatter_matrix(overall_mean, red_mean, blue_mean, purple_mean)

    max_eigenvectors = compute_max_eigenvectors(overall_within_scatter_matrix, overall_between_scatter_matrix)

    orange_points = transform_pts(red_points, max_eigenvectors)
    lightblue_points = transform_pts(blue_points, max_eigenvectors)
    lightpurple_points = transform_pts(purple_points, max_eigenvectors)

    print(f"transformed red points: {orange_points}")
    print(f"transformed blue points: {lightblue_points}")
    if NUM_OF_PURPLE_PTS != 0:
        print(f"transformed purple points: {lightpurple_points}")

    if BEG_DIMENSIONS == 2:
        fig, ax = update_plot_2d(fig, ax, orange_points, lightblue_points, lightpurple_points, max_eigenvectors[0])
    elif BEG_DIMENSIONS == 3:
        fig, ax = update_plot_3d(fig, ax, orange_points, lightblue_points, lightpurple_points, max_eigenvectors)
    plt.show()

if __name__ == '__main__':
    main()