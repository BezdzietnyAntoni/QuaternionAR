"""_summary_

    Returns:
        _type_: _description_
    """

import numpy as np
import quaternion as qt


def read_attitude_quaternions(log):
    """
    Read attitude from ulog file.
    Args:
        log (pyulog.ULog): PyUlog object to read.

    Returns:
        quat (np.ndarray [np.quaternion]), time_stamp (np.ndarray) :
        Attitude quaternion array, timestamp array
    """
    v_attitude = log.get_dataset('vehicle_attitude')
    data_size  = v_attitude.data['q[0]'].shape[0]
    merged     = zip(v_attitude.data['q[0]'], 
                     v_attitude.data['q[1]'],
                     v_attitude.data['q[2]'],
                     v_attitude.data['q[3]'])

    quat = np.zeros((data_size), dtype=np.quaternion)
    for i, m in enumerate(merged):
        quat[i] = np.quaternion(m[0], m[1], m[2], m[3])
    
    return quat, v_attitude.data['timestamp']


def read_nav_quaternions(log):
    """
    Read navigator from ulog file.
    Args:
        log (pyulog.ULog): PyUlog object to read.

    Returns:
        quat (np.ndarray [np.quaternion]), time_stamp (np.ndarray) :
        Navigator quaternion array, timestamp array
    """
    nav = log.get_dataset('navigator_setpoint')
    quat = qt.from_euler_angles(nav.data['sp_Roll_angle'], 
                                nav.data['sp_Pitch_angle'],
                                nav.data['sp_Yaw_angle'])
    return quat, nav.data['timestamp']