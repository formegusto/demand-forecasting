import numpy as np
from numpy import dot
from numpy.linalg import norm
import numpy as np
from scipy.spatial.distance import euclidean as euc


def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))


def euc_dis(A, B):
    return euc(A, B)


def calc_sin(ts, target_value):
    return np.sin(ts * (2 * np.pi / target_value)).values


def calc_cos(ts, target_value):
    return np.cos(ts * (2 * np.pi / target_value)).values


def get_season(month):
    if month in [3, 4, 5]:
        return "봄"
    elif month in [6, 7, 8]:
        return "여름"
    elif month in [9, 10, 11]:
        return "가을"
    else:
        return "겨울"


def get_season_to_idx(season):
    if season == "봄":
        return 1
    elif season == "여름":
        return 2
    elif season == "가을":
        return 3
    elif season == "겨울":
        return 4


SEASONS = ["봄", "여름", "가을", "겨울"]
