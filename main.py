import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from src.kinematics import setup_plot_axes, create_visualization_elements
from src.dynamics import run_dynamics_simulation
from src.parameters import Parameters
from src.simulation import RobotArmAnimation

def main():
    # Set plot style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10

    # Create parameter manager
    params = Parameters()

    # Create figure and 3D axes
    fig = plt.figure(figsize=(15, 8))

    # Create grid for subplots
    gs = plt.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)

    # Create 2D axes for animation
    ax = fig.add_subplot(gs[0])

    # Create text axes for parameter explanation
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')

    # Add parameter explanation
    explanation = """
    2D Two-Link Robot Arm Parameters:

    Physical Parameters:
    • L1, L2: Lengths of links 1 and 2 (meters)
        - Determines the arm's reach and workspace

    • m1, m2: Masses of links 1 and 2 (kg)
        - Affects the arm's inertia and dynamics

    • g: Gravitational acceleration (m/s²)
        - Points in negative Y direction (downward)
        - Earth's gravity = 9.81 m/s²

    • tau1, tau2: Joint torques (N⋅m)
        - τ1: Controls rotation of first link
        - τ2: Controls rotation of second link
        - Positive = counterclockwise

    Initial Conditions:
    • theta1_0, theta2_0: Initial joint angles (rad)
        - θ1: Angle of first link from X-axis
        - θ2: Relative angle between links

    • omega1_0, omega2_0: Initial angular velocities (rad/s)
        - ω1: First link angular velocity
        - ω2: Second link angular velocity

    Joint Configuration:
    - Joint 1 (Base): Anchored at origin
    - Joint 2 (Elbow): Between links

    Coordinate System:
    - Origin at base of the arm
    - Y-axis: vertical (gravity points down)
    - X-axis: horizontal
    """

    ax_text.text(0, 0.95, explanation,
                transform=ax_text.transAxes,
                verticalalignment='top',
                fontsize=10,
                fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Set up plot axes and create visualization elements
    setup_plot_axes(ax, params.params['L1'], params.params['L2'])
    plot_elements = create_visualization_elements(ax)

    # Create animation manager
    animation = RobotArmAnimation(fig, ax, plot_elements)

    # Add parameter sliders
    slider_ax_y = 0.02  # Starting y position for sliders
    slider_height = 0.02
    slider_spacing = 0.03

    # Create slider axes
    slider_axes = {}
    for i, (param, limits) in enumerate(params.PARAM_LIMITS.items()):
        y_pos = slider_ax_y + i * slider_spacing
        ax_pos = plt.axes([0.1, y_pos, 0.3, slider_height])
        slider_axes[param] = ax_pos

    # Create sliders
    sliders = {}
    for param, ax in slider_axes.items():
        min_val, max_val = params.PARAM_LIMITS[param]

        # Get initial value from parameters
        if param in params.params:
            initial_val = params.params[param]
        else:
            initial_val = params.initial_conditions[param]

        slider = Slider(
            ax, param,
            min_val, max_val,
            valinit=initial_val
        )
        sliders[param] = slider

        # Create update function for this parameter
        def make_update_func(param_name):
            def update(val):
                params.update_param(param_name, val)
                if param_name in ['L1', 'L2']:
                    # Update axes limits based on new arm lengths
                    setup_plot_axes(ax, params.params['L1'], params.params['L2'])
                    animation.init_animation()
                    fig.canvas.draw_idle()
            return update

        slider.on_changed(make_update_func(param))

    # Add simulation control button
    button_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
    sim_button = Button(button_ax, 'Simulate')

    def run_simulation(event):
        # Get current simulation parameters
        sim_params = params.sim_params
        physical_params = params.params
        initial_state = params.get_initial_state_vector()

        # Run simulation
        solution = run_dynamics_simulation(
            initial_state,
            physical_params,
            (sim_params['t_start'], sim_params['t_end']),
            sim_params['dt']
        )

        if solution.success:
            # Start animation with new solution
            animation.start_animation(solution, physical_params, sim_params['dt'])

    sim_button.on_clicked(run_simulation)

    # Adjust subplot parameters to make room for sliders
    plt.subplots_adjust(left=0.1, bottom=0.4)

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
