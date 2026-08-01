import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mujoco_py import mjviewer

mjviewer
# Parameters
T = 0.5
N_sim = 50

# initial and end condition

xi = np.array([0, 0]) # [x,y]
goal = np.array([5, 4])

# Obstacle
x_obs = 2.5
y_obs = 2.5
r_obs = 0.8

# CBF parameter
gain_alpha = 1.5

# controller
gain_nom_k = 1

# input limit
v_max = 0.5

# CBF function

def cbf(xi, x_obs, y_obs, r_obs):

    h = (xi[0] - x_obs)**2 + (xi[1] - y_obs)**2 - r_obs**2

    dh_dx = 2*(xi[0] - x_obs)
    dh_dy = 2*(xi[1] - y_obs)
    dh = np.array([dh_dx, dh_dy])

    fx = np.zeros(2, dtype= float)
    gx = np.identity(2, dtype = float)

    lf_h = dh @ fx
    lg_h = dh @ gx

    return h, lf_h, lg_h

def nom_controller(xi, goal, gain_nom_k, v_max):

    u_nom = -gain_nom_k*(xi - goal)

    speed = np.sum((u_nom)**2)

    return u_nom

def safety_filter(u_nom, xi, x_obs, y_obs, r_obs, gain_alpha, v_max):

    h, lf_h, lg_h = cbf(xi, x_obs, y_obs, r_obs)

    def cost(u):
        J = np.sum((u - u_nom)**2)
        return J

    cbf_con = { 'type' : 'ineq', 'fun' : lambda u: lf_h + lg_h @ u + gain_alpha * h}

    bounds = [(-v_max, v_max), (-v_max, v_max)]

    result = minimize(cost, u_nom, method='slsqp', constraints=cbf_con, bounds=bounds)

    return result.x

traj_nom = np.zeros((2, N_sim + 1))
traj_filt = np.zeros((2, N_sim + 1))
h_nom = np.zeros(N_sim)
h_filt = np.zeros(N_sim)

traj_nom[:,0] = xi
traj_filt[:,0] = xi

xi_nom = xi.copy()
for i in range(N_sim):

    u_nom = nom_controller(xi_nom, goal, gain_nom_k, v_max)

    h, lf_h, lg_h = cbf(xi_nom, x_obs, y_obs, r_obs)

    h_nom[i] = h

    xi_nom = xi_nom + T*u_nom

    traj_nom[:, i+1] = xi_nom

xi_filt = xi.copy()

for i in range(N_sim):
    u_nom = nom_controller(xi_filt, goal, gain_nom_k, v_max)
    u_safe = safety_filter(u_nom, xi_filt, x_obs, y_obs, r_obs, gain_alpha, v_max)

    h, lf_h, lg_h = cbf(xi_filt, x_obs, y_obs, r_obs)
    print(f"Step {i} | h={h:.3f} | u_nom={u_nom} | u_safe={u_safe}")

    h_filt[i] = h
    xi_filt = xi_filt + T * u_safe
    traj_filt[:, i + 1] = xi_filt

# Plot 1 — Trajectory
plt.subplot(1,2,1)
plt.plot(traj_nom[0, :], traj_nom[1, :], 'r--', label='Nominal')
plt.plot(traj_filt[0, :], traj_filt[1, :], 'b-', label='Filtered')
plt.plot(xi[0], xi[1], 'ro', markersize=10, label='Start')
plt.plot(goal[0], goal[1], 'go', markersize=10, label='Goal')
theta_draw = np.linspace(0, 2*np.pi, 100)
x_circle = x_obs + r_obs * np.cos(theta_draw)
y_circle = y_obs + r_obs * np.sin(theta_draw)
plt.fill(x_circle, y_circle, color='red', alpha=0.55, linewidth=2, label='Obstacle')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('CBF Safety Filter — Trajectory')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('')

# Plot 2 — h over time
plt.subplot(1,2,2)
plt.plot(h_nom,  'r--', label='h nominal')
plt.plot(h_filt, 'b-',  label='h filtered')
plt.axhline(y=0, color='k', linestyle='--', label='Safety boundary')
plt.xlabel('Time step')
plt.ylabel('h(x)')
plt.title('CBF Safety Margin Over Time')
plt.legend()
plt.grid(True)

plt.show()
plt.savefig('Safety_Filter_Margin_Over_Time.jpeg')



