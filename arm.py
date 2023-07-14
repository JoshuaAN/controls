import numpy as np
from math import *

def rk4(f, x, u, dt):
    """Performs 4th order Runge-Kutta integration of dx/dt = f(x, u) for dt.

    Keyword arguments:
    f -- vector function to integrate
    x -- vector of states
    u -- vector of inputs (constant for dt)
    dt -- time for which to integrate
    """
    half_dt = dt * 0.5
    k1 = f(x, u)
    k2 = f(x + half_dt * k1, u)
    k3 = f(x + half_dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def main():
    # Dynamics
    # 
    #   M(q)v̇ + C(q, v) = B(q)u
    # 
    #   q̇ = G(q)v
    # 
    #   ẋ = f(x, u) = [         G(q)v         ]
    #                 [M(q)⁻¹(B(q)u - C(q, v))]
    # 
    # Pendulum
    # 
    #   M(q) = m * l²
    #   C(q, v) = m * g * l * sin(θ)
    #   B(q) = I
    #   G(q) = I
    m = 1
    g = 9.81
    l = 1

    M = m * l * l
    B = 1
    G = 1

    T = 1
    dt = 0.005
    N = T/dt

    def f(x, u):
        q = x[0]
        v = x[1]

        C = m * g * l * sin(q)
        
        return np.array([[G * v, (B * u - C) / M]]).T
    
    def discrete_f(x, u):
        return rk4(f, x, u, dt)
    
    xs = np.zeros((2, N + 1))
    us = np.zeros((1, N))

    for iterate in range(100):
        # Forward rollout

        # Backward pass