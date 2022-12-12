import numpy as np
import quaternion as qt
from functools import singledispatch
from typing import Union


def quaternion_euler_difference(q1_arr: np.ndarray, q2_arr: np.ndarray):
    """
        Function calculate difference between quaternion in euler angles. 
    Args:
        q1_arr (np.ndarray): Array of quaternion as euler angles or quaternion 
        q2_arr (np.ndarray): Array of quaternion as euler angles or quaternion

    Returns:
        q_diff(np.ndarray): Difference between quaternion
    """
    if q1_arr.dtype == qt.quaternion:  
        q1_arr = quaternion_to_euler(q1_arr)

    if q2_arr.dtype == qt.quaternion:  
        q2_arr = quaternion_to_euler(q2_arr)
    
    q_diff = q1_arr - q2_arr
    q_diff = np.abs( (q_diff + 180) % 360 - 180 )
   
    return q_diff


def quaternion_euler_mse(q1_arr: np.ndarray, q2_arr: np.ndarray):
    """
        Return mse error for roll, pitch, yaw
    Args:
        q1_arr (np.ndarray): Array of quaternion as euler angles or quaternion 
        q2_arr (np.ndarray): Array of quaternion as euler angles or quaternion

    Returns:
        mse (np.ndarray): MSE error for components
    """
    
    diff = quaternion_euler_difference(q1_arr, q2_arr)
    mse = np.average(diff**2, axis=0)
    return mse


def quaternion_to_euler(q_arr: np.ndarray):
    """
        Casting quaternion to euler angles
    Args:
        q_arr (np.ndarray): Array with quaternions

    Returns:
        q_euler: Array of quaternion as euler angles in degrees
    """
    

    q_arr_f = qt.as_float_array(q_arr)
    q_euler = np.zeros((q_arr.shape[0], 3))

    # Pitch 
    t0 = 2. * (q_arr_f[:,0] * q_arr_f[:,1] + q_arr_f[:,2] * q_arr_f[:,3])
    t1 = 1. - 2. * (q_arr_f[:,1]**2 + q_arr_f[:,2]**2)
    q_euler[:,0] = np.degrees(np.arctan2(t0, t1))

    # Roll
    t2 = 2. * (q_arr_f[:,0] * q_arr_f[:,2] - q_arr_f[:,3] * q_arr_f[:,1])
    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    q_euler[:,1] = np.degrees(np.arcsin(t2))

    # Yaw
    t3 = 2. * (q_arr_f[:,0] * q_arr_f[:,3] + q_arr_f[:,1] * q_arr_f[:,2])
    t4 = +1.0 - 2.0 * (q_arr_f[:,2]**2 + q_arr_f[:,3]**2)
    q_euler[:,2] = np.degrees(np.arctan2(t3, t4))

    return q_euler