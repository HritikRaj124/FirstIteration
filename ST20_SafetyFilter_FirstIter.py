import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameters Simulation
T = 0.1
N_sim = 300

# Parameter Robot 1
xi_1 = np.array([-3,3])
goal_1 = np.array([3,3])

# Parameter Robot 2
xi_2 = np.array([3,3])
goal_2 = np.array([-3,3])

# CBF parameter
alpha_gain = 1
d_min = 0.5

# Nominal controller gain
k_gain = 1

v_max = 0.5

# CBF Function

def cbf_pair(xi_1, xi_2, d_min):

    h = ((xi_1[0] - xi_2[0])**2) + ((xi_1[1] - xi_2[1])**2) - d_min**2

    dh_dx1 = 2*(xi_1[0] - xi_2[0])
    dh_dy1 = 2*(xi_1[1] - xi_2[1])
    dh = np.array([dh_dx1, dh_dy1])

    fx = np.zeros(2, dtype= float)
    gx = np.identity(2, dtype = float)

    lf_h = dh @ fx
    lg_h = dh @ gx

    return h, lf_h, lg_h

def nom_controller(xi, goal, k_gain, v_max):

    u_nom = -k_gain*(xi - goal)

    speed = np.linalg.norm(u_nom)

    if speed > v_max:
        u_nom = v_max * u_nom / np.linalg.norm(u_nom)

    return u_nom

def safety_filter(u_nom, xi_self, xi_other, d_min, alpha_gain, v_max):

    h12, lf_h, lg_h = cbf_pair(xi_self, xi_other, d_min)

    def cost(u):
        J = np.linalg.norm(u - u_nom)**2
        return J

    cbf_cons = {'type': 'ineq', 'fun' : lambda u : lf_h + lg_h @ u + alpha_gain * h12}

    bounds = [(-v_max, v_max), (-v_max, v_max)]

    result = minimize(cost, u_nom, method='SLSQP', constraints=cbf_cons, bounds=bounds)
    return result.x

# storage
traj_nom_1 = np.zeros((2, N_sim+1))
traj_nom_2 = np.zeros((2, N_sim+1))
traj_filt_1 = np.zeros((2, N_sim+1))
traj_filt_2 = np.zeros((2, N_sim+1))
h_nom = np.zeros(N_sim)
h_filt = np.zeros(N_sim)
intervention_1 = np.zeros(N_sim)
intervention_2 = np.zeros(N_sim)

# Initial positions
traj_nom_1[:, 0]  = xi_1
traj_nom_2[:, 0]  = xi_2
traj_filt_1[:, 0] = xi_1
traj_filt_2[:, 0] = xi_2

# Copies for simulation
xi_nom_1  = xi_1.copy()
xi_nom_2  = xi_2.copy()
xi_filt_1 = xi_1.copy()
xi_filt_2 = xi_2.copy()


for i in range(N_sim):

    u_nom_1 = nom_controller(xi_nom_1, goal_1, k_gain, v_max )
    u_nom_2 = nom_controller(xi_nom_2, goal_2, k_gain, v_max)

    h, lf_h, lg_h = cbf_pair(xi_nom_1, xi_nom_2, d_min)
    h_nom[i] = h

    xi_nom_1 = xi_nom_1 + T * u_nom_1
    xi_nom_2 = xi_nom_2 + T * u_nom_2

    traj_nom_1[:, i+1] = xi_nom_1
    traj_nom_2[:, i + 1] = xi_nom_2

for i in range(N_sim):
    u_nom_1 = nom_controller(xi_filt_1, goal_1, k_gain, v_max)
    u_nom_2 = nom_controller(xi_filt_2, goal_2, k_gain, v_max)

    u_safe_1 = safety_filter(u_nom_1, xi_filt_1, xi_filt_2, d_min, alpha_gain, v_max)
    u_safe_2 = safety_filter(u_nom_2, xi_filt_2, xi_filt_1, d_min, alpha_gain, v_max)

    h, lf_h, lg_h = cbf_pair(xi_filt_1, xi_filt_2, d_min)
    #print(f"Step {i} | h={h:.3f} | u_nom={u_nom} | u_safe={u_safe}")

    h_filt[i] = h
    xi_filt_1 = xi_filt_1 + T * u_safe_1
    xi_filt_2 = xi_filt_2 + T * u_safe_2

    traj_filt_1[:, i + 1] = xi_filt_1
    traj_filt_2[:, i + 1] = xi_filt_2

    # After computing u_safe_1 and u_safe_2
if np.linalg.norm(u_safe_1 - u_nom_1) > 1e-4:
    intervention_1[i] = 1

if np.linalg.norm(u_safe_2 - u_nom_2) > 1e-4:
    intervention_2[i] = 1

plt.figure()
plt.step(range(N_sim), intervention_1, 'r*', label='Robot 1 intervention')
plt.step(range(N_sim), intervention_2, 'bo', label='Robot 2 intervention')
plt.xlabel('Time step')
plt.ylabel('Intervention (1=active)')
plt.title('CBF Intervention Diagnostics')
plt.legend()
plt.grid(True)
plt.savefig('intervention_ST20.png')

plt.figure()
plt.plot(traj_nom_1[0, :], traj_nom_1[1, :], 'r--', label='Robot 1 Nominal')
plt.plot(traj_nom_2[0, :], traj_nom_2[1, :], 'b*', label='Robot 2 Nominal')
plt.plot(traj_filt_1[0, :], traj_filt_1[1, :], 'ro', label='Robot 1 Filtered')
plt.plot(traj_filt_2[0, :], traj_filt_2[1, :], 'bs', label='Robot 2 Filtered')
plt.plot(goal_1[0], goal_1[1], 'r*', markersize=15, label='Goal 1')
plt.plot(goal_2[0], goal_2[1], 'b*', markersize=15, label='Goal 2')
plt.plot(xi_1[0], xi_1[1], 'rs', markersize=10, label='Start 1')
plt.plot(xi_2[0], xi_2[1], 'bs', markersize=10, label='Start 2')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Multi-Robot CBF Safety Filter')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('Trajectory_ST20.png')

plt.figure()
plt.plot(h_nom,  'r--', label='h nominal')
plt.plot(h_filt, 'b-',  label='h filtered')
plt.axhline(y=0, color='k', linestyle='--', label='Safety boundary (h=0)')
plt.axhline(y=d_min**2, color='r', linestyle=':', label='d_min threshold')
plt.xlabel('Time step')
plt.ylabel('h(x)')
plt.title('Pairwise Safety Margin Over Time')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('pairwise_distance_ST20.png')

# Liveness check
goal_threshold = 0.2

reached_1 = np.linalg.norm(xi_filt_1 - goal_1) < goal_threshold
reached_2 = np.linalg.norm(xi_filt_2 - goal_2) < goal_threshold

print(f"Robot 1 reached goal: {reached_1}")
print(f"Robot 2 reached goal: {reached_2}")
print(f"Minimum pairwise h (nominal):  {min(h_nom):.4f}")
print(f"Minimum pairwise h (filtered): {min(h_filt):.4f}")
print(f"Safety violated (nominal):  {any(h < 0 for h in h_nom)}")
print(f"Safety violated (filtered): {any(h < 0 for h in h_filt)}")
print(f"Robot 1 interventions: {int(sum(intervention_1))}/{N_sim} steps")
print(f"Robot 2 interventions: {int(sum(intervention_2))}/{N_sim} steps")





