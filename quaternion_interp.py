import numpy as np
import quaternion as qt
from quaternion.calculus import spline_evaluation
from quaternion import squad



def discontinued_subspaces(att_t: np.ndarray, max_length: int) -> list:
    """
    Function find range of subsequence of data where 
    separator is max length between samples.

    Args:
        att_t (np.ndarray): One dimension array of numbers data
        max_length (int): Maximum length between samples (separator)

    Returns:
        arg_range (list): Subspaces indices
    """
    diff = np.diff(att_t)

    arg_botom = 0
    arg_top = np.argmax(diff > max_length)

    range_value = list()
    while arg_top:
        range_value.append([arg_botom, arg_botom+arg_top])
        arg_botom += arg_top + 1
        arg_top = np.argmax(diff[arg_botom+1:] > max_length)
    
    range_value.append([arg_botom, att_t.shape[0]])
    return range_value


def normalized_timestamp(t_arr: np.ndarray, min: int, max: int):
    """
    Simply data normalization (`t_arr`-`min`)/(`max`-`min`)

    Args:
        t_arr (np.ndarray): Data to be normalized
        min (int): Minimum value `min`>=min(t_arr)
        max (int): Maximum value `max`<=max(t_arr)

    Returns:
        t_arr_norm (np.ndarray): Normalized data
    """
    return (t_arr - min) / (max - min)


def spline_interpolation(q_arr: np.ndarray, t_arr: np.array, t_interp_arr: np.ndarray, spline_degree: int = 1):
    """
    Interpolation quaternion by spline method

    Args:
        q_arr (np.ndarray): Quaternion data array 
        t_arr (np.array): Timestamp of `q_arr`
        t_interp_arr (np.ndarray): Timestamp of interpolated data, must be include in `t_arr` range value
        spline_degree (int, optional): Degree of spline method in range `1...5`. Defaults to 1.

    Returns:
        q_interp_arr (np.ndarray): Interpolated quaternions
    """
    if t_arr[0] > t_interp_arr[0]:
        print("Interpolated timestamp is smaller than the range of the data array")
        return
    
    if t_arr[-1] < t_interp_arr[-1]:
        print("Interpolated timestamp is bigger than the range of the data array")
        return

    # Normalization
    min_val = t_arr[0] 
    max_val = t_arr[-1]

    t_arr_norm = normalized_timestamp(t_arr, min_val, max_val)
    t_interp_arr_norm = normalized_timestamp(t_interp_arr, min_val, max_val)

    # Unpack quaternion
    q_arr_unpacked = qt.as_float_array(q_arr)

    # Interpolation
    q_interp_unpacked = spline_evaluation(q_arr_unpacked, t_arr_norm, t_interp_arr_norm, spline_degree=spline_degree)

    # Pack quaternion to np.quaternion and return
    return qt.from_float_array(q_interp_unpacked)


def squad_interpolation(q_arr: np.ndarray, t_arr: np.array, t_interp_arr: np.ndarray):
    """
    Interpolation quaternion by squad method, based on slerp method

    Args:
        q_arr (np.ndarray): Quaternion data array 
        t_arr (np.array): Timestamp of `q_arr`
        t_interp_arr (np.ndarray): Timestamp of interpolated data, must be include in `t_arr` range value

    Returns:
        q_interp_arr (np.ndarray): Interpolated quaternions
    """
    if t_arr[0] > t_interp_arr[0]:
        print("Interpolated timestamp is smaller than the range of the data array")
        return
    
    if t_arr[-1] < t_interp_arr[-1]:
        print("Interpolated timestamp is bigger than the range of the data array")
        return

    # Normalization
    min_val = t_arr[0] 
    max_val = t_arr[-1]

    t_arr_norm = normalized_timestamp(t_arr, min_val, max_val)
    t_interp_arr_norm = normalized_timestamp(t_interp_arr, min_val, max_val)

    # Interpolation
    q_interp = squad(q_arr, t_arr_norm, t_interp_arr_norm)

    # Pack quaternion to np.quaternion and return
    return q_interp