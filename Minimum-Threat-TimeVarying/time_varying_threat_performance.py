import numpy as np
import pandas as pd


from threat import threat_varying
from plotting import paper_plotting


def system_dynamics(state: np.ndarray, psi: float, t: float, spd: float):
    """
    Function which takes in the current state, time, and control input, and returns the value of the associated differential equations
    :param state: current state, first and second parameter must be x1 and x2
    :param psi: current heading angle
    :param t: current time
    :param spd: speed of the vehicle (constant)
    :return: vector containing x_dot, p_dot, and H_dot
    """
    c, c_x1, c_x2, c_t = threat_varying(state[0], state[1], t)

    # vehicle dynamics
    x_dot = spd * np.array([np.cos(psi), np.sin(psi)])
    # co-state dynamics
    p_dot = - np.array([c_x1, c_x2])
    # H_dot
    H_dot = np.array([c_t])

    return np.concatenate([x_dot, p_dot, H_dot])


def rk4(state0: np.ndarray, p0: np.ndarray, t_vec: np.ndarray, heading_angle: np.ndarray, spd: float = 1):
    """
    Function which takes an initial state & control input and determines the system variables
    :param state0: initial state (x1, x2)
    :param p0: guess for the co-state initial values (p1, p2)
    :param t_vec: time vector (covering the length of the trajectory)
    :param heading_angle: vector containing the heading angle for all timesteps (control input)
    :param spd: speed of the vehicle (constant)
    :return: array containing all system values: (x1, x2, p1, p2, H) for all timesteps (n by 5 vector)
    """
    system = np.zeros((len(t_vec), 5))  # create an empty vector to hold system variables
    dt = t_vec[1:] - t_vec[:-1]  # find dt for all timesteps
    # set initial conditions
    system[0, :2] = state0
    system[0, 2:4] = p0
    system[0, 4] = 0    # initial guess for the initial value of H (we shift this later)

    # integrate the system dynamics
    curr_state = system[0, :]
    for i, time in enumerate(t_vec[:-1]):
        # relevant heading angles
        psi1 = heading_angle[i]      # at t = t_i
        psi2 = heading_angle[i + 1]  # at t = t_i+1
        psi_mid = (psi1 + psi2) / 2  # approximation at the midpoint

        k1 = system_dynamics(curr_state, psi1, time, spd) * dt[i]
        k2 = system_dynamics(curr_state + k1 * dt[i] / 2, psi_mid, time + dt[i] / 2, spd) * dt[i]
        k3 = system_dynamics(curr_state + k2 * dt[i] / 2, psi_mid, time + dt[i] / 2, spd) * dt[i]
        k4 = system_dynamics(curr_state + k3 * dt[i], psi2, time + dt[i], spd) * dt[i]
        curr_state = curr_state + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        system[i + 1, :] = curr_state   # save the new state value
    return system


def residuals(psi: np.ndarray, t: np.ndarray, p_guess: np.ndarray, x0: np.ndarray, xf: np.ndarray, spd: float = 1):
    """
    Function to find residuals given an initial state of the system
    :param psi: heading angle along the trajectory (control input)
    :param t: time vector (corresponds to the heading angle)
    :param p_guess: guess for the initial value of the co-states
    :param x0: initial state (x1, x2)
    :param xf: desired final state
    :param spd: speed of the vehicle (constant)
    :return: residuals:
        - H - H_analytical  -> value of the Hamiltonian vs the "analytic" value (determine by integrating c_t and applying b.c.)
        - H_dot - H_dot_analytic    -> derivative of the Hamiltonian vs the desired value (c_t)
        - dH/du     -> derivative of the Hamiltonian w.r.t. to psi (should be 0 for this problem)
        - x(tf) - x_f   -> final state
    """
    # find (x1, x2, p1, p2, H)
    state_vec = rk4(x0, p_guess, t, psi, spd)
    x_vec = state_vec[:, :2]
    p_vec = state_vec[:, 2:4]
    H_analytic = state_vec[:, 4]

    # constraint on the Hamiltonian: H(tf) = 0
    H_shift = H_analytic[-1]    # find the value of H at the final time
    H_analytic = H_analytic - H_shift   # shift the Hamiltonian to satisfy the final b.c.

    # find the actual Hamiltonian (according to PMP)
    c, c_x1, c_x2, c_t = threat_varying(x_vec[:, 0], x_vec[:, 1], t)
    H = c + p_vec[:, 0] * spd * np.cos(psi) + p_vec[:, 1] * spd * np.sin(psi)

    # approximate psi_dot
    dudt = np.gradient(psi, t)
    # find H_dot analytically (from PMP eqn for H)
    H_dot = (c_t + c_x1 * spd * np.cos(psi) + c_x2 * spd * np.sin(psi) -
             c_x1 * spd * np.cos(psi) - p_vec[:, 0] * spd * np.sin(psi) * dudt -
             c_x2 * spd * np.sin(psi) + p_vec[:, 1] * spd * np.cos(psi) * dudt)

    # find dHdu analytically (from PMP eqn for H)
    dHdu = -p_vec[:, 0] * spd * np.sin(psi) + p_vec[:, 1] * spd * np.cos(psi)

    return H - H_analytic, H_dot - c_t, dHdu, x_vec[-1] - xf


def visualize_trajectory(psi: np.ndarray, t: np.ndarray, p_guess: np.ndarray, x0: np.ndarray, spd: float = 1):
    """
    Find all parameters of the system given a particular control input and initial state
    :param psi: heading angle along the trajectory (control input)
    :param t: time vector (corresponds to the heading angle)
    :param p_guess: guess for the initial value of the co-states
    :param x0: initial state (x1, x2)
    :param spd: speed of the vehicle (constant)
    :return: system parameters:
        - x1        -> x1 coordinate along the trajectory
        - x2        -> x2 coordinate
        - p1        -> p1 value
        - p2        -> p2 value
        - H_analytic -> desired value of the Hamiltonian (from integration & b.c.s)
        - H         -> actual value of the Hamiltonian (from PMP eqn)
        - c_t       -> desired value of dH/dt
        - H_dot     -> actual value of dH/dt
        - dHdu      -> derivative of the Hamiltonian w.r.t. psi (should be zero)
    """
    # find (x1, x2, p1, p2, H)
    state_vec = rk4(x0, p_guess, t, psi, spd)
    x_vec = state_vec[:, :2]
    p_vec = state_vec[:, 2:4]
    H_analytic = state_vec[:, 4]

    # constraint on the Hamiltonian: H(tf) = 0
    H_shift = H_analytic[-1]  # find the value of H at the final time
    H_analytic = H_analytic - H_shift  # shift the Hamiltonian to satisfy the final b.c.

    # find the actual Hamiltonian (according to PMP)
    c, c_x1, c_x2, c_t = threat_varying(x_vec[:, 0], x_vec[:, 1], t)
    H = c + p_vec[:, 0] * spd * np.cos(psi) + p_vec[:, 1] * spd * np.sin(psi)

    # approximate psi_dot
    dudt = np.gradient(psi, t)
    # find H_dot analytically (from PMP eqn for H)
    H_dot = (c_t + c_x1 * spd * np.cos(psi) + c_x2 * spd * np.sin(psi) -
             c_x1 * spd * np.cos(psi) - p_vec[:, 0] * spd * np.sin(psi) * dudt -
             c_x2 * spd * np.sin(psi) + p_vec[:, 1] * spd * np.cos(psi) * dudt)

    # find dHdu analytically (from PMP eqn for H)
    dHdu = -p_vec[:, 0] * spd * np.sin(psi) + p_vec[:, 1] * spd * np.cos(psi)

    return x_vec[:, 0], x_vec[:, 1], p_vec[:, 0], p_vec[:, 1], H_analytic, H, c_t, H_dot, dHdu


if __name__ == "__main__":
    # check function
    data = pd.read_csv('00-57-40-dynamic.csv')
    (x1_, x2_, p1_, p2_, H_d, H_, H_t_d, H_t, H_u) = visualize_trajectory(psi=data.psi.to_numpy(),
                                                                          t=data.time.to_numpy(),
                                                                          p_guess=np.array([data.p1.to_numpy()[0], data.p2.to_numpy()[0]]),
                                                                          x0=np.array([data.x0.to_numpy()[0], data.y0.to_numpy()[0]]),
                                                                          spd=10)
    paper_plotting(time=data.time.to_numpy(),
                   hamiltonian_NN=data.hamiltonian,
                   x1_NN=data.x1,
                   x2_NN=data.x2,
                   p1_NN=data.p1,
                   p2_NN=data.p2,
                   H_dot_NN=data.h_dot,
                   h_dot=H_t,
                   initial_state=[data.x0.to_numpy()[0], data.y0.to_numpy()[0]],
                   final_state=[data.xf.to_numpy()[0], data.yf.to_numpy()[0]],
                   p1=p1_,
                   p2=p2_,
                   dHdu_NN=data.control_constraint,
                   x1=x1_,
                   x2=x2_,
                   h_desired=H_d,
                   dHdu=H_u,
                   h=H_,
                   h_dot_desired=H_t_d)
