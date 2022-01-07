import sys
from importlib.resources import read_binary

from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BallWindow, BoxWindow

if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


def load_ginibre():
    # todo explain how data was generated
    return __load_point_pattern_in_ball("ginibre_data.pkl")


def load_kly():
    # todo explain how data was generated
    return __load_point_pattern_in_box("kly_data.pkl")


def load_poisson():
    # todo explain how data was generated
    return __load_point_pattern_in_box("poisson_data.pkl")


def load_thomas():
    return __load_point_pattern_in_box("thomas_data.pkl")


def load_z2():
    # todo explain how data was generatedx
    return __load_point_pattern_in_box("z2_data.pkl")


def __load_point_pattern_in_ball(file_name):

    data = read_binary("structure_factor.data", file_name)
    dict_data = pickle.loads(data)

    points = dict_data["points"]
    center = dict_data["center"]
    radius = dict_data["radius"]
    window = BallWindow(center, radius)
    intensity = dict_data.get("intensity", None)

    return PointPattern(points=points, window=window, intensity=intensity)


def __load_point_pattern_in_box(file_name):

    data = read_binary("structure_factor.data", file_name)
    dict_data = pickle.loads(data)

    points = dict_data["points"]
    bounds = dict_data["bounds"]
    window = BoxWindow(bounds)
    intensity = dict_data.get("intensity", None)
    intensity_parent = dict_data.get("intensity_parent", None)  # for thomas pp
    sigma = dict_data.get("sigma", None)  # for thomas pp
    params = {
        "intensity_parent": intensity_parent,
        "sigma": sigma,
    }  # params for thomas pp
    return PointPattern(points=points, window=window, intensity=intensity, **params)
