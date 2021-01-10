import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sir_model_dynamics(
    N=1000,
    I0=10,
    R0=0,
    beta=0.2,
    gamma=1./10,
    t=np.linspace(0, 160, 160)
):

    S0 = N - I0 - R0

# The SIR model differential equations.

    def deriv(y, t, N, beta, gamma):
        S, I, _ = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    y0 = S0, I0, R0

    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I, R


if __name__ == "__main__":
    t = np.linspace(0, 160, 160)
    N = 1000
    S, I, R = sir_model_dynamics(N=1000, beta=0.3, gamma=1/6, t=t)
# Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure()
    # plt.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    plt.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    plt.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    plt.plot(t, I + R, 'k', lw=2, label="Infected + Recovered")
    plt.xlabel("Days")
    plt.ylabel("% of population")
    plt.legend()
    plt.show()
