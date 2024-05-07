import numpy as np


def new_data(
    min_k, max_k, pk_data, covariance_data, n_poles
):  # Función para establecer las dimensiones de los datos en función de los límites de k
    for i in range(len(pk_data[:, 0:1])):
        if pk_data[:, 0:1][i] == min_k:
            n = i
        if pk_data[:, 0:1][i] == max_k:
            m = i
    s = int(len(covariance_data[0]) / n_poles)
    p_subfile = pk_data[:, 1:2][n : m + 1]
    if n_poles == 2:
        p = np.vstack((p_subfile, pk_data[:, 2:3][n : m + 1]))
    if n_poles == 3:
        p_1 = np.vstack((p_subfile, pk_data[:, 2:3][n : m + 1]))
        p = np.vstack((p_1, pk_data[:, 3:][n : m + 1]))
    print(
        "Dimensiones del vector de datos de multipolos: ", p.shape
    )  # Creamos un vector con los datos de los 3 multipolos (p0,p2,p4)
    if n_poles == 2:
        mask1 = covariance_data[n : m + 1, n : m + 1]
        mask2 = covariance_data[n : m + 1, n + s : m + 1 + s]
        mask4 = covariance_data[n + s : m + 1 + s, n : m + 1]
        mask5 = covariance_data[n + s : m + 1 + s, n + s : m + 1 + s]
        mask7 = covariance_data[n + (2 * s) : m + 1 + (2 * s), n : m + 1]
        mask8 = covariance_data[n + (2 * s) : m + 1 + (2 * s), n + s : m + 1 + s]
        h1 = np.hstack((mask1, mask2))
        h3 = np.hstack((mask4, mask5))
        h5 = np.hstack((mask7, mask8))
        final1 = np.vstack((h1, h3))
        final = np.vstack((final1, h5))
        new_covariance = final
    if n_poles == 3:
        mask1 = covariance_data[n : m + 1, n : m + 1]
        mask2 = covariance_data[n : m + 1, n + s : m + 1 + s]
        mask3 = covariance_data[n : m + 1, n + (2 * s) : m + 1 + (2 * s)]
        mask4 = covariance_data[n + s : m + 1 + s, n : m + 1]
        mask5 = covariance_data[n + s : m + 1 + s, n + s : m + 1 + s]
        mask6 = covariance_data[n + s : m + 1 + s, n + (2 * s) : m + 1 + (2 * s)]
        mask7 = covariance_data[n + (2 * s) : m + 1 + (2 * s), n : m + 1]
        mask8 = covariance_data[n + (2 * s) : m + 1 + (2 * s), n + s : m + 1 + s]
        mask9 = covariance_data[
            n + (2 * s) : m + 1 + (2 * s), n + (2 * s) : m + 1 + (2 * s)
        ]
        h1 = np.hstack((mask1, mask2))
        h2 = np.hstack((h1, mask3))
        h3 = np.hstack((mask4, mask5))
        h4 = np.hstack((h3, mask6))
        h5 = np.hstack((mask7, mask8))
        h6 = np.hstack((h5, mask9))
        final1 = np.vstack((h2, h4))
        final = np.vstack((final1, h6))
        new_covariance = final
    num_k = [min_k, max_k, (max_k - min_k) / (len(p) / n_poles)]
    print(
        "Las dimensiones de la matriz de covarianza son: ", new_covariance.shape
    )  # Creamos una matriz de covarianza nueva, eliminando los datos que exceden los límites de k
    return p, new_covariance, np.array(num_k)
