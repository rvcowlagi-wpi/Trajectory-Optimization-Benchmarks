import numpy as np

def threat_varying(x1, x2, t):
    threat_field = 0
    threat_gradient_x1 = 0
    threat_gradient_x2 = 0
    threat_gradient_t = 0

    n_peaks = 11
    coeff_peaks = np.array([[0.4218, 1.4253, 0.8271, 1.5330, 1.7610, 0.4533, 0.2392, 0.8364, 0.5060, 1.5190, 2],
                            [-4.7996, 6.8744, 1.6560, 4.3881, -3.1295, -9.8145, -6.1511, 0.1478, -9.5152, 1.0008, 9],
                            [-4.4763, 3.9248, 7.6271, -9.5064, -3.1759, -1.5719, -8.3998, -8.4129, -8.5525, 8.0068, -1],
                            [3.2579, 1.5239, 1.2908, 2.0099, 2.7261, 2.7449, 2.9398, 2.7439, 0.3691, 3.1097, 3],
                            [0.4039, 0.4382, 2.4844, 1.9652, 1.9238, 1.8567, 0.5470, 1.0401, 0.7011, 3.3193, 3]])
    const_1 = 5

    for m11 in range(n_peaks):
        c_xym = np.exp(
            - ((x1 - coeff_peaks[1, m11]) ** 2) / (2 * coeff_peaks[3, m11] ** 2)
            - ((x2 - coeff_peaks[2, m11]) ** 2) / (2 * coeff_peaks[4, m11] ** 2))
        threat_field = threat_field + (coeff_peaks[0, m11] / 2 * (np.cos(coeff_peaks[0, m11] * t) + 1.5)) * c_xym

        threat_gradient_x1 = (threat_gradient_x1 -
                              const_1 * ((x1 - coeff_peaks[1, m11]) / (coeff_peaks[3, m11] ** 2)) *
                              (coeff_peaks[0, m11] / 2 * (np.cos(coeff_peaks[0, m11] * t) + 1.5)) * c_xym)
        threat_gradient_x2 = (threat_gradient_x2 -
                              const_1 * ((x2 - coeff_peaks[2, m11]) / (coeff_peaks[4, m11] ** 2)) *
                              (coeff_peaks[0, m11] / 2 * (np.cos(coeff_peaks[0, m11] * t) + 1.5)) * c_xym)
        threat_gradient_t = (threat_gradient_t -
                             const_1 * coeff_peaks[0, m11]**2 / 2 * np.sin(coeff_peaks[0, m11] * t) * c_xym)

    threat_field = const_1 * threat_field + 1

    return threat_field, threat_gradient_x1, threat_gradient_x2, threat_gradient_t
