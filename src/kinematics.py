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

    First joint (th1) is angle from X-axis.
    Second joint (th2) is relative angle between links.

    Args:
        th1: First joint angle (rad) - angle from X-axis
        th2: Second joint angle (rad) - relative angle between links
        l1, l2: Link lengths (m)
    Returns:
        Tuple of (base_position, joint1_position, end_effector_position)
    """
    # Base is at origin
    base = np.array([0, 0])

    # First joint position
    joint1_pos = np.array([
        l1 * np.cos(th1),  # X
        l1 * np.sin(th1)   # Y
    ])

    # End effector position - add the second link's contribution
    end_effector_pos = np.array([
        l1 * np.cos(th1) + l2 * np.cos(th1 + th2),  # X
        l1 * np.sin(th1) + l2 * np.sin(th1 + th2)   # Y
    ])

    return base, joint1_pos, end_effector_pos

def setup_plot_axes(ax, l1, l2):
    """Sets up the static parts of the plot with correct orientation."""
    max_reach = l1 + l2
    margin = 0.3

    # Set axis limits
    ax.set_xlim([-max_reach - margin, max_reach + margin])
    ax.set_ylim([-max_reach - margin, max_reach + margin])

    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')

    # Setup grid
    ax.grid(True, alpha=0.2)

    # Set aspect ratio to equal for proper visualization
    ax.set_aspect('equal')

    # Set the title
    ax.set_title('2D Two-Link Robot Arm', fontsize=16, fontweight='bold', pad=20)

def create_visualization_elements(ax):
    """Creates and returns the plot elements that will be animated."""
    line_link1, = ax.plot([], [], color=COLORS['link1'], linewidth=10, solid_capstyle='round', label='Link 1')
    line_link2, = ax.plot([], [], color=COLORS['link2'], linewidth=10, solid_capstyle='round', label='Link 2')
    trace_line, = ax.plot([], [], color=COLORS['trace'], linestyle='--', linewidth=1, label='EE Trace')
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)

    ax.legend(loc='upper right')

    return {
        'line_link1': line_link1,
        'line_link2': line_link2,
        'trace_line': trace_line,
        'time_text': time_text
    }
