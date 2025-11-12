import matplotlib.pyplot as plt
import numpy as np

from threat import threat_varying

def paper_plotting(time, x1, x2, p1, p2, h, h_desired, h_dot, h_dot_desired, dHdu, initial_state, final_state,
                   x1_NN=None, x2_NN=None, p1_NN=None, p2_NN=None, hamiltonian_NN=None, H_dot_NN=None, dHdu_NN=None
                   ):
    """

    :param time: time of the trajectory
    :param x1: x1 coordinate
    :param x2: x2 coordinate
    :param p1: p1 value
    :param p2: p2 value
    :param h: hamiltonian
    :param h_desired: desired value (found by integration)
    :param h_dot: h_dot of the trajectory
    :param h_dot_desired: desired value (dc/dt)
    :param dHdu: derivative of H w.r.t. u
    :param initial_state: initial state (x1, x2)
    :param final_state: final state (x1, x2)
    :param x1_NN: x1 solution found via NN
    :param x2_NN: x2 solution found via NN
    :param p1_NN: p1 solution found via NN
    :param p2_NN: p2 solution found via NN
    :param hamiltonian_NN: hamiltonian for NN solution
    :param H_dot_NN: H_dot for NN soln
    :param dHdu_NN: dH/du for NN soln
    :return: creates a plot
    """
    plt.rcParams.update({
        "figure.figsize": (17, 8),  # Default figure size
        "axes.titlesize": 15,  # Title font size
        "axes.labelsize": 14,  # X and Y label size
        "xtick.labelsize": 10,  # X-axis tick labels
        "ytick.labelsize": 10,  # Y-axis tick labels
        "legend.fontsize": 10,  # Legend font size
        "lines.linewidth": 1.5,  # Thicker plot lines
        "lines.markersize": 6,  # Marker size
        "axes.grid": True,  # Enable grid
        "grid.alpha": 0.3,  # Transparent grid
        "grid.linestyle": "--",  # Dashed grid lines
        "axes.spines.top": False,  # Remove top spine
        "axes.spines.right": False,  # Remove right spine
        "legend.frameon": True,  # Remove legend box border
        "font.family": "Times New Roman",  # Use a clean, readable font
        "text.usetex": True
    })

    # Top row
    ax1 = plt.subplot2grid((2, 4), (0, 0))
    ax2 = plt.subplot2grid((2, 4), (0, 1))
    ax3 = plt.subplot2grid((2, 4), (0, 2))
    ax4 = plt.subplot2grid((2,4), (0, 3))

    # Bottom row, columns 0 to 3
    ax5 = plt.subplot2grid((2, 4), (1, 0))
    ax6 = plt.subplot2grid((2, 4), (1, 1))
    ax7 = plt.subplot2grid((2, 4), (1, 2))
    ax8 = plt.subplot2grid((2, 4), (1, 3))

    if hamiltonian_NN is not None:
        ax1.plot(time, hamiltonian_NN, label='Numeric Soln (NN)', color='black')
    ax1.plot(time, h_desired, label=r'$H(\mathbf{x}, \mathbf{p}, u, t)$', linestyle='dashdot',
             color='red')
    ax1.plot(time, h, label=r'$\int_{t_0}^{t_f} \frac{\partial c}{\partial t}dt - H(t_f)$', linestyle='dotted',
             color='gray')
    ax1.scatter(time[-1], 0, s=50, marker='*', facecolors='gray',
                edgecolors='gray', label='Boundary Condition')
    ax1.set_ylabel('H')
    ax1.set_xlabel('Time [s]')
    ax1.set_title('Hamiltonian')
    ax1.legend()

    if H_dot_NN is not None:
        ax2.plot(time, H_dot_NN, label='Numeric Soln (NN)', color='black')
    ax2.plot(time, h_dot, label=r'$\dot{H}(\mathbf{x}, \mathbf{p}, u, t)$', linestyle='dashdot', color='red')
    ax2.plot(time, h_dot_desired, label=r'$\frac{\partial c}{\partial t}$', linestyle='dotted', color='gray')
    ax2.set_ylabel(r'$\dot{H}$')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('Hamiltonian Derivative')
    ax2.legend()

    x2_field = np.linspace(-15, 15, 1000)
    min_val = min(x1_NN)
    if min_val > -15:
        min_val = -15
    x1_field = np.linspace(min_val, 15, 1000)
    X, Y = np.meshgrid(x1_field, x2_field)
    threat_vals, grad_x1, grad_x2, grad_t = threat_varying(X, Y, time[-1])

    ax3.contourf(X, Y, threat_vals, levels=250)
    ax3.plot(x1, x2, linewidth=1,
             linestyle='-', color='gray', label='Analytic Path')
    if x1_NN is not None and x2_NN is not None:
        ax3.plot(x1_NN, x2_NN, linewidth=1,
                 linestyle='dashdot', color='white', label='PINN Path')
    ax3.scatter(initial_state[0], initial_state[1], s=50, marker='*', facecolors='red',
                edgecolors='red')
    ax3.scatter(final_state[0], final_state[1], s=40, marker='o', facecolors='red',
                edgecolors='red')
    ax3.set_title('Position')
    ax3.set_xlabel(r'$x_1$ [m]')
    ax3.set_ylabel(r'$x_2$ [m]')
    ax3.set_xlim([min_val, 15])
    ax3.legend()

    if dHdu_NN is not None:
        ax4.plot(time, dHdu_NN, label='Numeric Soln (NN)', color='black')
    ax4.plot(time, dHdu, label=r'$\frac{dH}{du}$', color='red', linestyle='dashdot')
    ax4.plot([time[0], time[-1]], [0, 0], label='Desired Value', color='gray', linestyle='dotted')
    ax4.set_title(r'Minimum Hamiltonian Condition')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel(r'$\frac{dH}{du}$')
    ax4.legend()

    if p1_NN is not None:
        ax5.plot(time, p1_NN, label=r'Numeric Soln (NN)', color='black')
    ax5.plot(time, p1, linestyle='dashdot', color='red',
             label=r'$p_1$')
    ax5.legend()
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel(r'${p_1}$')
    ax5.set_title('Co-State')

    if p2_NN is not None:
        ax6.plot(time, p2_NN, label=r'Numeric Soln (NN)', color='black')
    ax6.plot(time, p2, linestyle='dashdot', color='red',
             label=r'$p_2$')
    ax6.legend()
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel(r'${p_2}$')
    ax6.set_title('Co-State')

    if x1_NN is not None:
        ax7.plot(time, x1_NN, label=r'Numeric Soln (NN)', color='black')
    ax7.plot(time, x1, color='red',
             linestyle='dashdot', label=r'$x_1$')
    ax7.legend()
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel(r'${x_1}$ [m]')
    ax7.set_title('State')

    if x2_NN is not None:
        ax8.plot(time, x2_NN, label=r'Numeric Soln (NN)', color='black')
    ax8.plot(time, x2, color='red',
             linestyle='dashdot', label=r'$x_2$')
    ax8.legend()
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel(r'${x_2}$ [m]')
    ax8.set_title('State')

    plt.tight_layout()
    plt.show()
