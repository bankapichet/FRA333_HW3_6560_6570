import numpy as np
from HW3_utils import FKHW3
def endEffectorJacobianHW3(q):
    R, P, R_e, p_e = FKHW3(q)
    
    J_v = np.zeros((3, 3))
    J_w = np.zeros((3, 3))

    # Loop through each joint to calculate the Jacobian components
    for i in range(3):
        # Get the rotation matrix and position of each joint
        R_i = R[:, :, i]
        P_i = P[:, i]
        
        # Calculate linear velocity Jacobian
        z = R_i @ np.array([0, 0, 1])  # Axis of rotation for each joint
        J_v[:, i] = np.cross(z, (p_e - P_i))  # Cross product
        
        # Calculate angular velocity Jacobian
        J_w[:, i] = z

    # Combine linear and angular velocity Jacobian
    J_e = np.vstack((J_v, J_w))
    return J_e
