import numpy as np
np.set_printoptions(precision=6, suppress=True)


def generate_edges(n, phi):
    """
    Construct the polyhedral cone with angle phi and n edges.
    :param n: <int> Number of cone edges
    :param phi: <float> Cone angle
    :return: <2-dim np.array> Cone edge matrix of size (n, 3)

    Test cases:
    >>> generate_edges(phi=0.4, n=5)
    array([[ 0.921061,  0.      ,  0.389418],
           [ 0.284624,  0.875981,  0.389418],
           [-0.745154,  0.541386,  0.389418],
           [-0.745154, -0.541386,  0.389418],
           [ 0.284624, -0.875981,  0.389418]])
    >>> generate_edges(phi=np.pi/4, n=3)
    array([[ 0.707107,  0.      ,  0.707107],
           [-0.353553,  0.612372,  0.707107],
           [-0.353553, -0.612372,  0.707107]])
    """
    # ------------------------------------------------
    cone_edges = np.empty([n, 3])
    for i in range(n):
        theta = 2 * np.pi * i / n
        cone_edges[i] = [np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), np.sin(phi)]
    # ------------------------------------------------
    assert isinstance(cone_edges, np.ndarray), 'Wrong return type for generate_edges. Make sure it is a np.ndarray'
    return cone_edges


def compute_normals(cone_edges):
    """
    Compute the facet normals given the cone edge matrix.
    :param cone_edges: <2-dim np.array> Cone edge matrix of size (n, 3)
    :return: <2-dim np.array> Facet normals matrix of size (n, 3)

    Test cases:
    >>> compute_normals(np.array([[ 0.70710678,  0.        ,  0.70710678],[-0.35355339,  0.61237244,  0.70710678], [-0.35355339, -0.61237244,  0.70710678]]))
    array([[-0.433013, -0.75    ,  0.433013],
           [ 0.866025,  0.      ,  0.433013],
           [-0.433013,  0.75    ,  0.433013]])
    """
    # ------------------------------------------------
    facet_normals = np.cross(cone_edges, np.roll(cone_edges, -1, axis=0))
    # ------------------------------------------------
    assert isinstance(facet_normals, np.ndarray), 'Wrong return type for compute_normals. Make sure it is a np.ndarray'
    return facet_normals


def compute_minimum_distance_from_facet_normals(a, facet_normals):
    """
    Compute the minimum distance from a point 'a' to the polyhedral
    cone parametrized by the given facet normals.
    :param a: <np.array> 3D point
    :param facet_normals: <2-dim np.array> Facet normals matrix of size (n, 3)
    :return: <float> Minimum distance from 'a' to the cone.

    Test cases:
    >>> abs(compute_minimum_distance_from_facet_normals(a=np.array([0,0,1]), facet_normals=np.array([[-4.33012702e-01, -7.50000000e-01,  4.33012702e-01], [ 8.66025404e-01, -3.33066907e-16,  4.33012702e-01], [-4.33012702e-01,  7.50000000e-01,  4.33012702e-01]]))-0.4472135954999579)<0.00001
    True
    >>> abs(compute_minimum_distance_from_facet_normals(a=np.array([10,-10,0.2]), facet_normals=np.array([[-4.33012702e-01, -7.50000000e-01,  4.33012702e-01], [ 8.66025404e-01, -3.33066907e-16,  4.33012702e-01], [-4.33012702e-01,  7.50000000e-01,  4.33012702e-01]]))-3.3632734550894954)<0.00001
    True
    """
    # ------------------------------------------------
    normalized_facet_normals = facet_normals / np.linalg.norm(facet_normals, ord=2, axis=1, keepdims=True)
    minimum_distance = min(np.abs(np.matmul(normalized_facet_normals, a)))
    # ------------------------------------------------
    assert isinstance(minimum_distance, float), 'Wrong return type for compute_minimum_distance_from_facet_normals. Make sure it is a float'
    return minimum_distance


def compute_minimum_distance(a, n, phi):
    """
    Compute the minimum distance from a point 'a' to the polyhedral
    cone of n edges and angle phi
    :param a: <np.array> 3D point
    :param n: <int> Number of cone edges
    :param phi: <float> Cone angle
    :return: <float> Minimum distance from 'a' to the cone.

    Test cases:
    >>> abs(compute_minimum_distance(a=np.array([0.2,-0.3,0.1]), n=7, phi=0.3)-0.0062024061079438446)<0.000001
    True
    >>> abs(compute_minimum_distance(a=np.array([0.2,-0.3,0.1]), n=10, phi=0.01)-0.0962065349510294)<0.000001
    True
    """
    # ------------------------------------------------
    cone_edges = generate_edges(n, phi)
    facet_normals = compute_normals(cone_edges)
    minimum_distance = compute_minimum_distance_from_facet_normals(a, facet_normals)
    # ------------------------------------------------
    assert isinstance(minimum_distance, float), 'Wrong return type for compute_minimum_distance_from_facet_normals. Make sure it is a float'
    return minimum_distance


def check_is_interior_point(a, n, phi):
    """
    Return whether a is an interior point of the polyhedral cone
    of n edges and angle phi
    :param a: <np.array> 3D point
    :param n: <int> Number of cone edges
    :param phi: <float> Cone angle
    :return: <bool> If a is an interior point

    Test cases:
    >>> check_is_interior_point(a=np.array([0.2,-0.3,0.1]), n=7, phi=0.3)
    False
    >>> check_is_interior_point(a=np.array([0.2,-0.3,10]), n=7, phi=0.3)
    True
    """
    # ------------------------------------------------
    cone_edges = generate_edges(n, phi)
    facet_normals = compute_normals(cone_edges)

    is_interior_point = np.min(np.matmul(facet_normals, a)) > 0
    # ------------------------------------------------
    assert isinstance(is_interior_point, bool), 'Wrong return type for check_is_interior_point. Make sure it is a bool'
    return is_interior_point


if __name__ == "__main__":
    # Automated Testing:
    import doctest

    # Run tests cases for all functions:
    # doctest.testmod(verbose=True) # Uncomment to test all functions

    # Tests for only a desired function (uncomment the one for the function to test):
    # doctest.run_docstring_examples(generate_edges, globals(), verbose=True)   # Uncomment to test generate_edges
    # doctest.run_docstring_examples(compute_normals, globals(), verbose=True)  # Uncomment to test compute_normals
    # doctest.run_docstring_examples(compute_minimum_distance_from_facet_normals, globals(), verbose=True)  # Uncomment to test compute_minimum_distance_from_facet_normals
    # doctest.run_docstring_examples(compute_minimum_distance, globals(), verbose=True) # Uncomment to test compute_minimum_distance
    # doctest.run_docstring_examples(check_is_interior_point, globals(), verbose=True)  # Uncomment to test check_is_interior_point