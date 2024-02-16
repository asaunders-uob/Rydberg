import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy import constants
from scipy.special import diric
from scipy.integrate import quad

def poly_log(q, n, z):
    return (q**(n-1))/(np.exp(q) / z - 1)

def polylog(n, z):
    integrals = np.array([])
    for i in z:
        integral = quad(poly_log, 0, np.inf, args=(n,i))[0]
        integrals = np.append(integrals, integral)
    return integrals / gamma(n)
