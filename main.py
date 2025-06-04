import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Set clean styling
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Initialize parameters
L1 = 2.0  # Length of first link
L2 = 1.5  # Length of second link
theta1 = 0.0  # First joint angle (radians)
theta2 = 0.0  # Second joint angle (radians)

# Clean color scheme
COLORS = {
    'link1': '#2E86AB',
    'link2': '#A23B72',
    'base': '#F18F01',
    'joint': '#C73E1D',
    'end_effector': '#F18F01'
}

# Create figure
fig = plt.figure(figsize=(12, 9), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

def forward_kinematics(th1, th2, l1, l2):
    base = np.array([0, 0, 0])
    joint1 = np.array([l1 * np.cos(th1), l1 * np.sin(th1), 0])
    end_effector = np.array([
        l1 * np.cos(th1) + l2 * np.cos(th1 + th2),
        l1 * np.sin(th1) + l2 * np.sin(th1 + th2),
        0
    ])
    return base, joint1, end_effector

def update_arm():
    ax.clear()

    # Calculate positions
    base, joint1, end_effector = forward_kinematics(theta1, theta2, L1, L2)

    # Plot arm links
    ax.plot([base[0], joint1[0]], [base[1], joint1[1]], [base[2], joint1[2]],
            color=COLORS['link1'], linewidth=10, solid_capstyle='round', label='Link 1')

    ax.plot([joint1[0], end_effector[0]], [joint1[1], end_effector[1]], [joint1[2], end_effector[2]],
            color=COLORS['link2'], linewidth=10, solid_capstyle='round', label='Link 2')

    # Plot joints
    ax.scatter(*base, color=COLORS['base'], s=300, alpha=0.9, edgecolors='white', linewidth=3)
    ax.scatter(*joint1, color=COLORS['joint'], s=200, alpha=0.9, edgecolors='white', linewidth=2)
    ax.scatter(*end_effector, color=COLORS['end_effector'], s=200, alpha=0.9, edgecolors='white', linewidth=2)

    # Set axis properties
    max_reach = L1 + L2
    margin = 0.3
    ax.set_xlim([-max_reach-margin, max_reach+margin])
    ax.set_ylim([-max_reach-margin, max_reach+margin])
    ax.set_zlim([-0.5, 0.5])

    # Clean axis labels
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')

    # Clean grid
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

    # Set title
    ax.set_title('2DoF Robotic Arm', fontsize=16, fontweight='bold', pad=20)

    # Set viewing angle
    ax.view_init(elev=15, azim=45)

# Initial plot
update_arm()

# Create clean sliders
plt.subplots_adjust(bottom=0.25)

# Slider properties
slider_props = dict(facecolor=COLORS['link1'], alpha=0.6)

# Joint angle sliders
ax_theta1 = plt.axes([0.15, 0.15, 0.7, 0.03])
slider_theta1 = Slider(ax_theta1, 'θ1', -np.pi, np.pi, valinit=theta1,
                       valfmt='%.2f rad', **slider_props)

ax_theta2 = plt.axes([0.15, 0.10, 0.7, 0.03])
slider_theta2 = Slider(ax_theta2, 'θ2', -np.pi, np.pi, valinit=theta2,
                       valfmt='%.2f rad', **slider_props)

# Link length sliders
ax_l1 = plt.axes([0.15, 0.05, 0.3, 0.03])
slider_l1 = Slider(ax_l1, 'L1', 0.5, 3.0, valinit=L1,
                   valfmt='%.1f', **slider_props)

ax_l2 = plt.axes([0.55, 0.05, 0.3, 0.03])
slider_l2 = Slider(ax_l2, 'L2', 0.5, 3.0, valinit=L2,
                   valfmt='%.1f', **slider_props)

# Update functions
def update_theta1(val):
    global theta1
    theta1 = slider_theta1.val
    update_arm()
    update_info()
    plt.draw()

def update_theta2(val):
    global theta2
    theta2 = slider_theta2.val
    update_arm()
    update_info()
    plt.draw()

def update_l1(val):
    global L1
    L1 = slider_l1.val
    update_arm()
    update_info()
    plt.draw()

def update_l2(val):
    global L2
    L2 = slider_l2.val
    update_arm()
    update_info()
    plt.draw()

# Connect sliders
slider_theta1.on_changed(update_theta1)
slider_theta2.on_changed(update_theta2)
slider_l1.on_changed(update_l1)
slider_l2.on_changed(update_l2)

# Simple info display
def update_info():
    _, _, end_effector = forward_kinematics(theta1, theta2, L1, L2)

    info_text = f"End Effector Position: ({end_effector[0]:.2f}, {end_effector[1]:.2f}, {end_effector[2]:.2f})"

    # Update or create info text
    if hasattr(update_info, 'text_obj'):
        update_info.text_obj.set_text(info_text)
    else:
        update_info.text_obj = plt.figtext(0.5, 0.02, info_text, ha='center',
                                          fontsize=11, bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor='lightgray', alpha=0.8))

# Initial info update
update_info()

plt.tight_layout()
plt.show()