import numpy as np

def gen_rotation_matrix(alpha: float, phi: float) -> np.ndarray:
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    ca2 = np.cos(alpha / 2)
    sa2 = np.sin(alpha / 2)

    return np.array([
        [
            ca2 * ca2,
            sa2 * sa2 * np.exp(1j * 2.0 * phi),
            -1j * np.exp(1j * phi) * sa
        ],
        [
            np.exp(-1j * 2.0 * phi) * sa2 * sa2,
            ca2 * ca2,
            1j * np.exp(-1j * phi) * sa
        ],
        [-1j / 2 * np.exp(-1j * phi) * sa, 1j / 2 * np.exp(1j * phi) * sa, ca]
    ])
