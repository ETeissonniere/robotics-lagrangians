import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

from equations import theta1_lambda, theta2_lambda, eex_lambda, eey_lambda

# --- Configuration ---
L1 = 0.4  # Length of first link (m)
L2 = 0.3  # Length of second link (m)
M1 = 2.0  # Mass of first link (kg)
M2 = 1.5  # Mass of second link (kg)

I1 = (1/3) * M1 * L1**2 # Moment of inertia for link 1
I2 = (1/3) * M2 * L2**2 # Moment of inertia for link 2

G = 9.81  # Gravitational acceleration (m/s^2)

DT = 0.05 # Time step (s) - increased for a simpler trajectory
TOTAL_TIME = 6.0 # Total time for the simple trajectory (s)

# --- Forward Kinematics (Calculates end-effector X,Y from joint angles) ---
def forward_kinematics(theta1, theta2, l1, l2):
    params = [l1, l2, theta1, theta2]
    x = eex_lambda(*params)
    y = eey_lambda(*params)
    return x, y

# --- Simple Joint Space Trajectory Generation ---
def generate_simple_joint_trajectory(total_time, dt):
    time_vec = np.arange(0, total_time, dt)

    # Simple oscillating motion for theta1 and theta2
    # Theta1: moves from ~30 deg to ~120 deg and back
    theta1_traj = np.pi/6 + (np.pi/2) * (1 - np.cos(2 * np.pi * time_vec / total_time)) / 2
    # Theta2: moves from ~-90 deg to ~0 deg
    theta2_traj = -np.pi/2 + (np.pi/2) * (1 + np.sin(np.pi * time_vec / total_time)) / 2

    # Calculate velocities (theta_dot) using finite differences
    theta1_dot_traj = np.gradient(theta1_traj, dt)
    theta2_dot_traj = np.gradient(theta2_traj, dt)

    # Calculate accelerations (theta_double_dot) using finite differences
    theta1_ddot_traj = np.gradient(theta1_dot_traj, dt)
    theta2_ddot_traj = np.gradient(theta2_dot_traj, dt)

    return time_vec, theta1_traj, theta2_traj, theta1_dot_traj, theta2_dot_traj, theta1_ddot_traj, theta2_ddot_traj

# --- Calculate Joint Torques ---
def calculate_torques(time_vec, th1, th2, th1d, th2d, th1dd, th2dd):
    tau1_traj = np.zeros_like(time_vec)
    tau2_traj = np.zeros_like(time_vec)

    for i in range(len(time_vec)):
        # Parameters for torque equations:
        # m1, m2, l1, l2, i1, i2, g, th1, th1d, th1dd, th2, th2d, th2dd
        params = [M1, M2, L1, L2, I1, I2, G,
                  th1[i], th1d[i], th1dd[i],
                  th2[i], th2d[i], th2dd[i]]
        tau1_traj[i] = theta1_lambda(*params)
        tau2_traj[i] = theta2_lambda(*params)
    return tau1_traj, tau2_traj

# --- Main Animation Setup ---
fig, (ax_arm, ax_torque) = plt.subplots(1, 2, figsize=(12, 5))
trail_length = 50
end_effector_trail_x = deque(maxlen=trail_length)
end_effector_trail_y = deque(maxlen=trail_length)

# Arm plot setup
ax_arm.set_xlim(-(L1 + L2) * 1.1, (L1 + L2) * 1.1)
ax_arm.set_ylim(-(L1 + L2) * 0.2, (L1 + L2) * 1.1) # Adjusted y-limits slightly
ax_arm.set_aspect('equal', adjustable='box')
ax_arm.set_title('Robot Arm Motion')
ax_arm.set_xlabel('X (m)')
ax_arm.set_ylabel('Y (m)')
ax_arm.grid(True)
link1_plot, = ax_arm.plot([], [], 'b-', linewidth=5, label='Link 1')
link2_plot, = ax_arm.plot([], [], 'r-', linewidth=4, label='Link 2')
joint_base_plot, = ax_arm.plot(0, 0, 'ko', markersize=8, label='Base')
joint_elbow_plot, = ax_arm.plot([], [], 'go', markersize=6, label='Elbow')
end_effector_plot, = ax_arm.plot([], [], 'mo', markersize=7, label='End-Effector')
trail_plot, = ax_arm.plot([], [], 'c--', alpha=0.7, label='EE Trail')
ax_arm.legend(loc='upper right', fontsize='small')
floor_line = ax_arm.axhline(0, color='saddlebrown', lw=2) # Floor line

# Torque plot setup
torque_time_line, = ax_torque.plot([], [], 'b-', label='Joint 1 Torque')
torque_time_line2, = ax_torque.plot([], [], 'r-', label='Joint 2 Torque')
torque_progress_line = ax_torque.axvline(0, color='k', linestyle='--', lw=1)
ax_torque.set_title('Joint Torques')
ax_torque.set_xlabel('Time (s)')
ax_torque.set_ylabel('Torque (Nm)')
ax_torque.grid(True)
ax_torque.legend(fontsize='small')

# --- Animation Function ---
def animate(i, time_vec, th1_traj, th2_traj, tau1_plot_data, tau2_plot_data):
    theta1 = th1_traj[i]
    theta2 = th2_traj[i]

    # Link positions
    x0, y0 = 0, 0
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2, y2 = forward_kinematics(theta1, theta2, L1, L2)

    link1_plot.set_data([x0, x1], [y0, y1])
    link2_plot.set_data([x1, x2], [y1, y2])
    joint_elbow_plot.set_data([x1], [y1])
    end_effector_plot.set_data([x2], [y2])

    end_effector_trail_x.append(x2)
    end_effector_trail_y.append(y2)
    trail_plot.set_data(list(end_effector_trail_x), list(end_effector_trail_y))

    # Update torque plot data up to current time
    current_time = time_vec[i]
    torque_time_line.set_data(time_vec[:i+1], tau1_plot_data[:i+1])
    torque_time_line2.set_data(time_vec[:i+1], tau2_plot_data[:i+1])
    torque_progress_line.set_xdata([current_time, current_time])

    # Dynamically adjust torque plot limits if needed (optional)
    if i > 10 : # Start adjusting after a few points
        min_tau = min(np.min(tau1_plot_data[:i+1]), np.min(tau2_plot_data[:i+1]))
        max_tau = max(np.max(tau1_plot_data[:i+1]), np.max(tau2_plot_data[:i+1]))
        ax_torque.set_ylim(min_tau - abs(min_tau*0.1) - 0.5, max_tau + abs(max_tau*0.1) + 0.5) # Add some padding
        ax_torque.set_xlim(0, TOTAL_TIME)


    return link1_plot, link2_plot, joint_elbow_plot, end_effector_plot, trail_plot, \
           torque_time_line, torque_time_line2, torque_progress_line

# --- Generate Data & Run ---
print("🚀 Generating simple trajectory...")
time, th1, th2, th1_dot, th2_dot, th1_ddot, th2_ddot = generate_simple_joint_trajectory(TOTAL_TIME, DT)

print("💪 Calculating torques...")
tau1, tau2 = calculate_torques(time, th1, th2, th1_dot, th2_dot, th1_ddot, th2_ddot)

print("✨ Starting animation...")
# Set initial torque plot limits
if len(tau1) > 0 and len(tau2) > 0:
    ax_torque.set_ylim(min(np.min(tau1), np.min(tau2)) - 1, max(np.max(tau1), np.max(tau2)) + 1)
ax_torque.set_xlim(0, TOTAL_TIME)


ani = animation.FuncAnimation(fig, animate, frames=len(time),
                              fargs=(time, th1, th2, tau1, tau2),
                              interval=DT*1000, blit=True, repeat=True)

plt.tight_layout()
plt.show()

print("✅ Simulation finished.")
