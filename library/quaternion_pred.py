import numpy as np
import quaternion as qt


def compute_correlation_matrix(windowing_signal: np.ndarray, M_order: int) -> np.ndarray :
    """
    Determines the quaternion correlation matrix.
    Args:
        windowing_signal (np.ndarray[np.quaternion]): Windowing signal
        M_order (int): Filter order

    Returns:
        np.ndarray[np.quaternion]: Corelation matrix 
    """
    n = windowing_signal.size
    r_size = M_order + 1

    r_matrix = np.zeros((r_size, r_size), dtype = np.quaternion)
    for i in range(0, (n - M_order - 1) ):
        v = windowing_signal[i : (i + M_order + 1)]
        r_matrix += np.outer(v, v.conjugate())
    
    return r_matrix


def compute_coefficient_LD(r_matrix: np.ndarray, M_order: int) -> np.ndarray:
    """
        Compute coefficient using Levinson-Durbin method.   
    Args:
        r_matrix (np.ndarray): Correlation matrix
        M_order (int): Number of order 

    Returns:
        a_arr (np.ndarray): Coefficients
    """
    
    a_arr = np.zeros((0), dtype = np.quaternion)

    # Step 1: P=0 -> a0 = 1
    # Step 2: P=1
    a_arr = np.append(a_arr, r_matrix[1, 0] / r_matrix[0, 0])

    # Step 3: Iterative for P in (1 : M_order)
    for i in range(2, M_order + 1):
        nominator   = r_matrix[i, 0] - np.sum(a_arr * r_matrix[i,1:i])
        denominator = r_matrix[i, i]   - np.sum(a_arr[::-1] * r_matrix[i,1:i])
        temp_a = nominator / denominator

        # Coefficients actualization 
        a_arr = a_arr - temp_a * a_arr[::-1]

        # Add coefficient 
        a_arr = np.append(a_arr, temp_a)

    return a_arr