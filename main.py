import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque # For the trail

from equations import theta1_lambda, theta2_lambda

# ===== CONFIGURATION =====
class Config:
    def __init__(self):
        # Physics
        self.g = 9.81  # Gravitational acceleration (m/s^2)

        # Robot arm physical parameters
        self.m1 = 2.0  # Mass of first link (kg)
        self.m2 = 1.5  # Mass of second link (kg)
        self.l1 = 0.4  # Length of first link (m)
        self.l2 = 0.3  # Length of second link (m)
        self.i1 = (1/3) * self.m1 * self.l1**2
        self.i2 = (1/3) * self.m2 * self.l2**2

        # Trajectory waypoints (Pick and Place)
        self.home_pos = np.array([0.4, 0.25])
        self.pickup_pos = np.array([0.5, 0.12])
        self.lift_pos = np.array([0.5, 0.35])
        self.basket_pos = np.array([0.25, 0.12])

        # Motion timing (Pick and Place)
        self.segment_durations = {
            'home_to_pickup': 2.5, 'pickup_to_lift': 1.5,
            'lift_to_basket': 2.5, 'basket_to_home': 2.0
        }

        # Control parameters
        self.dt = 0.02
        self.max_joint_acceleration = 25.0
        self.singularity_threshold = 1e-6

        # Smoothing parameters
        self.cartesian_smooth_window = 7
        self.angle_smooth_window = 3
        self.trail_length = 75

        # Animation parameters
        self.animation_interval = int(self.dt * 1000)
        self.animation_repeat = True



# ===== ROBOT ARM =====
class RobotArm:
    def __init__(self, config: Config):
        self.config = config
        self.l1 = config.l1
        self.l2 = config.l2

    def forward_kinematics(self, theta1, theta2):
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        return x, y

    def inverse_kinematics(self, x_target, y_target, prev_theta1=None, prev_theta2=None):
        x, y = x_target, y_target
        r_sq = x**2 + y**2

        # Check reachability and adjust target if necessary to stay within workspace
        # This is a simplified adjustment, ensures the target is mathematically solvable
        eps = 0.001 # Small epsilon to avoid exact boundary
        max_reach = self.l1 + self.l2 - eps
        min_reach = abs(self.l1 - self.l2) + eps

        r = np.sqrt(r_sq)
        if r > max_reach:
            scale = max_reach / r
            x *= scale
            y *= scale
            r_sq = max_reach**2
        elif r < min_reach and r > 1e-3:
            scale = min_reach / r
            x *= scale
            y *= scale
            r_sq = min_reach**2
        elif r <= 1e-3:
             # Default to safe configuration for targets very close to origin
            if prev_theta1 is not None: return prev_theta1, prev_theta2
            return np.pi/2, -np.pi/2


        cos_theta2 = (r_sq - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)

        theta2_sol1 = np.arccos(cos_theta2)
        theta2_sol2 = -np.arccos(cos_theta2)

        solutions = []
        for th2_s in [theta2_sol1, theta2_sol2]:
            k1 = self.l1 + self.l2 * np.cos(th2_s)
            k2 = self.l2 * np.sin(th2_s)
            th1_s = np.arctan2(y, x) - np.arctan2(k2, k1)

            # Check floor constraint for elbow
            y_elbow = self.l1 * np.sin(th1_s)
            if y_elbow >= -0.005:
                solutions.append((th1_s, th2_s))

        if not solutions:
            # Fallback: pick the solution with higher elbow
            y_elbow1 = self.l1 * np.sin(np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(theta2_sol1), self.l1 + self.l2 * np.cos(theta2_sol1)))
            y_elbow2 = self.l1 * np.sin(np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(theta2_sol2), self.l1 + self.l2 * np.cos(theta2_sol2)))
            if y_elbow1 > y_elbow2:
                 th1_s = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(theta2_sol1), self.l1 + self.l2 * np.cos(theta2_sol1))
                 return th1_s, theta2_sol1
            else:
                 th1_s = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(theta2_sol2), self.l1 + self.l2 * np.cos(theta2_sol2))
                 return th1_s, theta2_sol2


        if prev_theta1 is not None and prev_theta2 is not None:
            best_sol = min(solutions, key=lambda sol: (sol[0] - prev_theta1)**2 + (sol[1] - prev_theta2)**2)
            return best_sol

        # Prefer "elbow up" configuration
        for sol in solutions:
            if sol[1] >= 0: return sol
        return solutions[0]


    def get_link_positions(self, theta1, theta2):
        x_j1, y_j1 = 0.0, 0.0
        x_j2 = self.l1 * np.cos(theta1)
        y_j2 = self.l1 * np.sin(theta1)
        x_ee, y_ee = self.forward_kinematics(theta1, theta2)
        return (x_j1, y_j1), (x_j2, y_j2), (x_ee, y_ee)

    def validate_floor_constraint(self, theta1_traj, theta2_traj):
        violations = 0
        for i, (th1, th2) in enumerate(zip(theta1_traj, theta2_traj)):
            y_elbow = self.l1 * np.sin(th1)
            _, y_ee = self.forward_kinematics(th1, th2)
            if y_elbow < -0.005 or y_ee < -0.005:
                violations +=1
                if violations < 5:
                     print(f"  WARN: Floor violation at step {i}: ElbowY={y_elbow:.3f}, EE_Y={y_ee:.3f}")
        if violations > 0:
            print(f"WARNING: {violations} total violations of floor constraint found!")
        else:
            print("✓ Arm trajectory respects floor constraints.")
        return violations == 0


# ===== TRAJECTORY UTILITIES =====
class TrajectoryUtils:
    @staticmethod
    def smooth_cubic_segment(start_pos, end_pos, duration, dt):
        time_pts = np.arange(0, duration, dt)
        if not time_pts.size: return np.array([]), np.array([]), np.array([]), np.array([])

        positions, velocities, accelerations = [], [], []
        for t in time_pts:
            s = t / duration

            blend = s**2 * (3 - 2*s)
            blend_dot = (6*s - 6*s**2) / duration
            blend_ddot = (6 - 12*s) / duration**2

            pos = start_pos + blend * (end_pos - start_pos)
            vel = blend_dot * (end_pos - start_pos)
            acc = blend_ddot * (end_pos - start_pos)
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)

        # Ensure last point is exactly end_pos
        if positions:
            positions[-1] = end_pos


        return time_pts, np.array(positions), np.array(velocities), np.array(accelerations)

    @staticmethod
    def smooth_moving_average(data, window_size):
        if window_size < 2 or len(data) < window_size: return data
        if isinstance(data, list): data = np.array(data)

        smoothed = np.copy(data)
        half_window = window_size // 2
        for i in range(half_window, len(data) - half_window):
            smoothed[i] = np.mean(data[i-half_window : i+half_window+1], axis=0)
        return smoothed

    @staticmethod
    def finite_difference_derivatives(positions, dt):
        if len(positions) < 2:
            return np.zeros_like(positions), np.zeros_like(positions)

        velocities = np.zeros_like(positions)
        accelerations = np.zeros_like(positions)

        velocities[1:] = (positions[1:] - positions[:-1]) / dt
        velocities[0] = velocities[1]

        accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt
        accelerations[0] = accelerations[1]

        return velocities, accelerations


# ===== TRAJECTORY PLANNER =====
class TrajectoryPlanner:
    def __init__(self, config: Config):
        self.config = config

    def generate_trajectory(self):
        return self._generate_pick_and_place_trajectory()

    def _generate_pick_and_place_trajectory(self):
        cfg = self.config
        waypoints = [cfg.home_pos, cfg.pickup_pos, cfg.lift_pos, cfg.basket_pos, cfg.home_pos]
        durations = [cfg.segment_durations['home_to_pickup'], cfg.segment_durations['pickup_to_lift'],
                       cfg.segment_durations['lift_to_basket'], cfg.segment_durations['basket_to_home']]

        full_time, full_pos, full_vel, full_acc = [], [], [], []
        current_time = 0.0

        for i in range(len(waypoints) - 1):
            t_seg, p_seg, v_seg, a_seg = TrajectoryUtils.smooth_cubic_segment(
                waypoints[i], waypoints[i+1], durations[i], cfg.dt
            )
            if t_seg.size == 0: continue

            full_time.append(t_seg + current_time)
            full_pos.append(p_seg)
            full_vel.append(v_seg)
            full_acc.append(a_seg)
            current_time += durations[i]

        time = np.concatenate(full_time) if full_time else np.array([])
        positions = np.vstack(full_pos) if full_pos else np.array([[],[]]).T

        # Initial smoothing and derivative recalculation
        if positions.shape[0] > cfg.cartesian_smooth_window:
            positions = TrajectoryUtils.smooth_moving_average(positions, cfg.cartesian_smooth_window)

        velocities, accelerations = TrajectoryUtils.finite_difference_derivatives(positions, cfg.dt)

        return time, positions, velocities, accelerations




# ===== DYNAMICS CALCULATOR =====
class DynamicsCalculator:
    def __init__(self, config: Config, robot_arm: RobotArm):
        self.config = config
        self.robot_arm = robot_arm

    def cartesian_to_joint_space(self, cartesian_pos, cartesian_vel, cartesian_acc):
        cfg = self.config
        num_pts = len(cartesian_pos)

        theta1_traj = np.zeros(num_pts)
        theta2_traj = np.zeros(num_pts)
        theta1d_traj = np.zeros(num_pts)
        theta2d_traj = np.zeros(num_pts)
        theta1dd_traj = np.zeros(num_pts)
        theta2dd_traj = np.zeros(num_pts)

        prev_th1, prev_th2 = None, None

        for i in range(num_pts):
            x, y = cartesian_pos[i]
            vx, vy = cartesian_vel[i]


            th1, th2 = self.robot_arm.inverse_kinematics(x, y, prev_th1, prev_th2)
            theta1_traj[i], theta2_traj[i] = th1, th2
            prev_th1, prev_th2 = th1, th2

            # Jacobian
            J = np.array([
                [-self.robot_arm.l1*np.sin(th1) - self.robot_arm.l2*np.sin(th1+th2), -self.robot_arm.l2*np.sin(th1+th2)],
                [ self.robot_arm.l1*np.cos(th1) + self.robot_arm.l2*np.cos(th1+th2),  self.robot_arm.l2*np.cos(th1+th2)]
            ])

            try:
                if abs(np.linalg.det(J)) > cfg.singularity_threshold:
                    joint_vel = np.linalg.solve(J, cartesian_vel[i])
                    theta1d_traj[i], theta2d_traj[i] = joint_vel[0], joint_vel[1]
                else:
                    theta1d_traj[i] = theta1d_traj[i-1] if i > 0 else 0
                    theta2d_traj[i] = theta2d_traj[i-1] if i > 0 else 0
            except np.linalg.LinAlgError:
                theta1d_traj[i] = theta1d_traj[i-1] if i > 0 else 0
                theta2d_traj[i] = theta2d_traj[i-1] if i > 0 else 0

        # Smooth joint angles
        if num_pts > cfg.angle_smooth_window:
            theta1_traj = TrajectoryUtils.smooth_moving_average(theta1_traj, cfg.angle_smooth_window)
            theta2_traj = TrajectoryUtils.smooth_moving_average(theta2_traj, cfg.angle_smooth_window)

        # Recalculate joint velocities and accelerations from smoothed joint angles using finite differences
        # Velocities
        theta1d_traj_fd, theta2d_traj_fd = TrajectoryUtils.finite_difference_derivatives(np.vstack((theta1_traj, theta2_traj)).T, cfg.dt)
        theta1d_traj = theta1d_traj_fd[:,0]
        theta2d_traj = theta1d_traj_fd[:,1]


        theta1dd_traj_fd, theta2dd_traj_fd = TrajectoryUtils.finite_difference_derivatives(np.vstack((theta1d_traj, theta2d_traj)).T, cfg.dt)
        theta1dd_traj = theta1dd_traj_fd[:,0]
        theta2dd_traj = theta1dd_traj_fd[:,1]


        # Clip joint accelerations
        theta1dd_traj = np.clip(theta1dd_traj, -cfg.max_joint_acceleration, cfg.max_joint_acceleration)
        theta2dd_traj = np.clip(theta2dd_traj, -cfg.max_joint_acceleration, cfg.max_joint_acceleration)

        return theta1_traj, theta2_traj, theta1d_traj, theta2d_traj, theta1dd_traj, theta2dd_traj

    def calculate_joint_torques(self, th1_t, th2_t, th1d_t, th2d_t, th1dd_t, th2dd_t):
        cfg = self.config
        tau1_traj, tau2_traj = [], []
        for i in range(len(th1_t)):
            params = [
                cfg.m1, cfg.m2, cfg.l1, cfg.l2, cfg.i1, cfg.i2, cfg.g,
                th1_t[i], th1d_t[i], th1dd_t[i],
                th2_t[i], th2d_t[i], th2dd_t[i]
            ]
            tau1_traj.append(theta1_lambda(*params))
            tau2_traj.append(theta2_lambda(*params))
        return np.array(tau1_traj), np.array(tau2_traj)


# ===== ROBOT ANIMATOR =====
class RobotAnimator:
    def __init__(self, config: Config, robot_arm: RobotArm, time_vec,
                 theta1_traj, theta2_traj, tau1_traj, tau2_traj, cartesian_traj_pts):
        self.config = config
        self.robot_arm = robot_arm
        self.time_vec = time_vec
        self.theta1_traj = theta1_traj
        self.theta2_traj = theta2_traj
        self.tau1_traj = tau1_traj
        self.tau2_traj = tau2_traj
        self.cartesian_traj_pts = cartesian_traj_pts # Full path for plotting

        self.fig, (self.ax_arm, self.ax_torque) = plt.subplots(1, 2, figsize=(14, 6))
        self.trail_x = deque(maxlen=config.trail_length)
        self.trail_y = deque(maxlen=config.trail_length)

        self._setup_arm_plot()
        self._setup_torque_plot()

    def _setup_arm_plot(self):
        ax = self.ax_arm
        cfg = self.config
        ax.set_xlim(-0.1, cfg.l1 + cfg.l2 + 0.1)
        ax.set_ylim(-0.1, cfg.l1 + cfg.l2 + 0.1)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.4)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Robot Arm: Pick And Place')

        ax.axhline(y=0, color='saddlebrown', linewidth=3, label='Floor', zorder=0)
        if self.cartesian_traj_pts.size > 0:
             ax.plot(self.cartesian_traj_pts[:,0], self.cartesian_traj_pts[:,1], 'g--', lw=1.5, alpha=0.6, label='Planned EE Path')

        ax.plot(cfg.pickup_pos[0], cfg.pickup_pos[1], 'rs', ms=8, label='Object', zorder=1)
        ax.plot(cfg.basket_pos[0], cfg.basket_pos[1], 'bs', ms=10, label='Basket', zorder=1)


        self.link1_plot, = ax.plot([], [], 'b-', linewidth=7, label='Link 1', zorder=2)
        self.link2_plot, = ax.plot([], [], 'r-', linewidth=5, label='Link 2', zorder=2)
        self.joint_base_plot, = ax.plot(0, 0, 'ko', markersize=10, label='Base', zorder=3)
        self.joint_elbow_plot, = ax.plot([], [], 'bo', markersize=7, zorder=3)
        self.end_effector_plot, = ax.plot([], [], 'ro', markersize=8, label='End Effector', zorder=3)
        self.trail_plot, = ax.plot([], [], 'lime', linewidth=2, alpha=0.7, label='EE Trail', zorder=1)
        ax.legend(loc='upper left', fontsize='small')

    def _setup_torque_plot(self):
        ax = self.ax_torque
        ax.plot(self.time_vec, self.tau1_traj, 'b-', lw=2, label='Joint 1 Torque')
        ax.plot(self.time_vec, self.tau2_traj, 'r-', lw=2, label='Joint 2 Torque')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (N·m)')
        ax.set_title('Required Joint Torques')
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize='small')
        self.progress_line = ax.axvline(x=0, color='k', linestyle='--', alpha=0.7)

    def _animate_frame(self, frame_idx):
        if frame_idx >= len(self.theta1_traj):
            # Handle cases where frame_idx might exceed trajectory length
            return (self.link1_plot, self.link2_plot, self.joint_elbow_plot,
                    self.end_effector_plot, self.trail_plot, self.progress_line)

        th1, th2 = self.theta1_traj[frame_idx], self.theta2_traj[frame_idx]
        (x_j1, y_j1), (x_j2, y_j2), (x_ee, y_ee) = self.robot_arm.get_link_positions(th1, th2)

        self.link1_plot.set_data([x_j1, x_j2], [y_j1, y_j2])
        self.link2_plot.set_data([x_j2, x_ee], [y_j2, y_ee])
        self.joint_elbow_plot.set_data([x_j2], [y_j2])
        self.end_effector_plot.set_data([x_ee], [y_ee])

        self.trail_x.append(x_ee)
        self.trail_y.append(y_ee)
        self.trail_plot.set_data(list(self.trail_x), list(self.trail_y))

        if self.time_vec.size > 0:
            self.progress_line.set_xdata([self.time_vec[frame_idx], self.time_vec[frame_idx]])

        return (self.link1_plot, self.link2_plot, self.joint_elbow_plot,
                self.end_effector_plot, self.trail_plot, self.progress_line)

    def run_animation(self):
        if not self.time_vec.size:
            print("No time vector. Cannot create animation.")

            if self.theta1_traj.size > 0:
                self._animate_frame(0)
            plt.tight_layout()
            plt.show()
            return None

        num_frames = len(self.time_vec)
        anim = animation.FuncAnimation(self.fig, self._animate_frame, frames=num_frames,
                                       interval=self.config.animation_interval,
                                       blit=True, repeat=self.config.animation_repeat)
        plt.tight_layout()
        plt.show()
        return anim


# ===== MAIN SCRIPT =====
def main():
    print("=== 2-DOF Robot Arm Simulation ===")
    config = Config()



    robot = RobotArm(config)
    planner = TrajectoryPlanner(config)
    calculator = DynamicsCalculator(config, robot)

    print("1. Generating Cartesian trajectory...")
    time_vec, cart_pos, cart_vel, cart_acc = planner.generate_trajectory()

    if time_vec.size == 0 or cart_pos.size == 0:
        print("ERROR: Trajectory generation failed or produced an empty trajectory.")
        return

    print(f"   Generated {len(time_vec)} points, duration: {time_vec[-1]:.2f}s")

    print("2. Converting to joint space and calculating derivatives...")
    th1, th2, th1d, th2d, th1dd, th2dd = calculator.cartesian_to_joint_space(
        cart_pos, cart_vel, cart_acc
    )

    print("3. Validating floor constraints for joint trajectory...")
    robot.validate_floor_constraint(th1, th2)

    print("4. Calculating joint torques...")
    tau1, tau2 = calculator.calculate_joint_torques(th1, th2, th1d, th2d, th1dd, th2dd)

    print("\n=== Motion Statistics ===")
    print(f"Total Duration: {time_vec[-1]:.2f} seconds")
    print(f"Max Joint 1 Angle: {np.degrees(np.max(np.abs(th1))):.1f}°")
    print(f"Max Joint 2 Angle: {np.degrees(np.max(np.abs(th2))):.1f}°")
    print(f"Max Joint 1 Torque: {np.max(np.abs(tau1)):.2f} N·m")
    print(f"Max Joint 2 Torque: {np.max(np.abs(tau2)):.2f} N·m")

    print("\n5. Creating animation...")
    animator = RobotAnimator(config, robot, time_vec, th1, th2, tau1, tau2, cart_pos)

    _ = animator.run_animation()
    print("Animation window opened. Close window to exit.")


if __name__ == "__main__":
    main()
