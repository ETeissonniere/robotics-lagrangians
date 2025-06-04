import numpy as np

# Colors for visualization
COLORS = {
    'link1': '#E74C3C',       # Red
    'link2': '#2ECC71',       # Green
    'trace': '#BDC3C7'        # Light gray
}

def forward_kinematics(th1, th2, l1, l2):
    """
    Calculate positions of the 2-DoF arm joints given angles and link lengths.

    First joint (th1) rotates around Z-axis, creating motion in X-Y plane.
    Second joint (th2) rotates around the X-axis of link 1's frame, creating up/down motion.

    Args:
        th1: First joint angle (rad) - rotation around Z
        th2: Second joint angle (rad) - rotation around link1's X axis
        l1, l2: Link lengths (m)
    Returns:
        Tuple of (base_position, joint1_position, end_effector_position)
    """
    # Base is at origin
    base = np.array([0, 0, 0])

    # First joint: rotate around Z axis
    joint1_pos = np.array([
        l1 * np.cos(th1),  # X
        l1 * np.sin(th1),  # Y
        0                  # Z
    ])

    # Second joint: first rotate with th1 around Z, then rotate with th2 around the new X axis
    # This creates a rotation in the YZ plane of the first link's frame
    end_effector_pos = np.array([
        l1 * np.cos(th1) + l2 * np.cos(th1),                    # X: same as joint1 X direction
        l1 * np.sin(th1) + l2 * np.cos(th2) * np.sin(th1),     # Y: project onto original Y
        l2 * np.sin(th2)                                        # Z: up/down motion
    ])

    return base, joint1_pos, end_effector_pos

def setup_plot_axes(ax, l1, l2):
    """Sets up the static parts of the plot with correct orientation."""
    max_reach = l1 + l2
    margin = 0.3

    # Set axis limits
    ax.set_xlim([-max_reach - margin, max_reach + margin])
    ax.set_ylim([-max_reach - margin, max_reach + margin])
    ax.set_zlim([-max_reach - margin, max_reach + margin])

    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')

    # Setup grid and pane properties
    ax.grid(True, alpha=0.2)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    # Set the title
    ax.set_title('2DoF Robotic Arm Dynamics', fontsize=16, fontweight='bold', pad=20)

    # Set view angle for better 3D visualization
    ax.view_init(elev=20, azim=45)

def create_visualization_elements(ax):
    """Creates and returns the plot elements that will be animated."""
    line_link1, = ax.plot([], [], [], color=COLORS['link1'], linewidth=10, solid_capstyle='round', label='Link 1')
    line_link2, = ax.plot([], [], [], color=COLORS['link2'], linewidth=10, solid_capstyle='round', label='Link 2')
    trace_line, = ax.plot([], [], [], color=COLORS['trace'], linestyle='--', linewidth=1, label='EE Trace')
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)

    ax.legend(loc='upper right')

    return {
        'line_link1': line_link1,
        'line_link2': line_link2,
        'trace_line': trace_line,
        'time_text': time_text
    }
