import numpy as np
import quaternion as qt
import matplotlib.pyplot as plt


def display_quaternion():
    pass


def display_quaternion_compare(
    q_arrays: list, 
    t_arrays: list, 
    labels: list = None, 
    title: str = ""):

    # Convert to w, x, y, z representation 
    for i, q_arr in enumerate(q_arrays):
        q_arrays[i] = qt.as_float_array(q_arr)

    # Plot section
    TITLES = ["quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"]
    plt.figure(figsize = (10,9))
    plt.tight_layout(pad = 5)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        for q_arr, t_arr, l in zip(q_arrays, t_arrays, labels):
            plt.plot(t_arr, q_arr[:,i], label=l) 
        plt.ylabel("Value")
        plt.xlabel("Timestamp")
        plt.title(TITLES[i])
        plt.legend(loc='upper right')

    plt.suptitle(title)
    plt.show()  

    