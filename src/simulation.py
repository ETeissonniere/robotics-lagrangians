from matplotlib.animation import FuncAnimation

from .kinematics import forward_kinematics

class RobotArmAnimation:
    def __init__(self, fig, ax, plot_elements):
        self.fig = fig
        self.ax = ax
        self.elements = plot_elements
        self.animation = None
        self.ee_trace = {'x': [], 'y': []}

    def init_animation(self):
        """Initialize animation elements."""
        self.ee_trace = {'x': [], 'y': []}

        # Reset all plot elements
        self.elements['line_link1'].set_data([], [])
        self.elements['line_link2'].set_data([], [])
        self.elements['trace_line'].set_data([], [])
        self.elements['time_text'].set_text('')

        return tuple(self.elements.values())

    def animate_frame(self, i, solution, params):
        """Update the plot for frame i of the animation."""
        th1 = solution.y[0, i]
        th2 = solution.y[1, i]
        current_time = solution.t[i]

        # Calculate new positions
        base_pos, joint1_pos, ee_pos = forward_kinematics(
            th1, th2,
            params['L1'],
            params['L2']
        )

        # Update link positions
        self.elements['line_link1'].set_data(
            [base_pos[0], joint1_pos[0]],
            [base_pos[1], joint1_pos[1]]
        )
        self.elements['line_link2'].set_data(
            [joint1_pos[0], ee_pos[0]],
            [joint1_pos[1], ee_pos[1]]
        )

        # Update end-effector trace
        self.ee_trace['x'].append(ee_pos[0])
        self.ee_trace['y'].append(ee_pos[1])
        self.elements['trace_line'].set_data(
            self.ee_trace['x'],
            self.ee_trace['y']
        )

        # Update time display
        self.elements['time_text'].set_text(f'Time: {current_time:.2f}s')

        return tuple(self.elements.values())

    def start_animation(self, solution, params, dt):
        """Start a new animation with the given solution data."""
        # Stop any existing animation
        if self.animation is not None:
            self.animation.event_source.stop()

        # Initialize and start new animation
        self.init_animation()
        self.animation = FuncAnimation(
            self.fig,
            self.animate_frame,
            frames=len(solution.t),
            fargs=(solution, params),
            init_func=self.init_animation,
            blit=True,
            interval=dt * 1000  # Convert to milliseconds
        )

        self.fig.canvas.draw_idle()
        return self.animation
