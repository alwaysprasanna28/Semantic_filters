import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class CBFRealtimePlot:
    def __init__(self, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), rotate_90=True):
        """
        Initializes the Real-time CBF Visualizer.
        
        Args:
            x_range: Tuple (min, max) for the plot X-axis (after rotation)
            y_range: Tuple (min, max) for the plot Y-axis (after rotation)
            rotate_90: If True, rotates the view 90 degrees counter-clockwise 
                       (World X -> Plot Y, World Y -> Plot -X).
                       Useful to align 'Robot Forward' with 'Screen Up'.
        """
        self.rotate_90 = rotate_90
        
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        # Axis setup
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        self.ax.set_xlabel("Plot Y (m)")
        self.ax.set_ylabel("Plot X (m)")
        title = "Top-Down CBF Safety Filter"
        self.ax.set_title(title)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_aspect('equal')

        # --- Visual Elements ---
        
        # 1. The Superquadric Obstacle (Red Zone)
        self.sq_patch = None 
        
        # 2. End Effector Position (Blue Dot)
        self.eef_dot, = self.ax.plot([], [], 'bo', markersize=8, label='EEF Position', zorder=5)
        
        # 3. Velocity Vectors
        # Raw Command (Red Arrow)
        self.quiver_raw = self.ax.quiver(0, 0, 0, 0, color='red', scale=1, scale_units='xy', angles='xy', label='Raw Cmd', width=0.015, zorder=6)
        # Safe Command (Green Arrow)
        self.quiver_safe = self.ax.quiver(0, 0, 0, 0, color='green', scale=1, scale_units='xy', angles='xy', label='Safe Cmd', width=0.015, zorder=7)

        self.ax.legend(loc='upper right')
        self.fig.tight_layout()

    def _transform(self, x, y):
        """
        Applies coordinate rotation if enabled.
        Standard 90 deg rotation: x' = -y, y' = x
        """
        if self.rotate_90:
            return -y, x
        return x, y

    def _transform_vector(self, vx, vy):
        """
        Applies rotation to direction vectors.
        """
        if self.rotate_90:
            return -vy, vx
        return vx, vy

    def _get_superquadric_contour(self, center, scale, epsilon, num_points=100):
        """
        Generates 2D (XY) polygon points for the superquadric boundary g(x)=1.
        Equation: (|x|/ax)^(2/e2) + (|y|/ay)^(2/e2) = 1
        """
        cx, cy = center[0], center[1]
        ax, ay = scale[0], scale[1]
        e2 = epsilon[1] # standard usually uses epsilon2 for XY curvature

        t = np.linspace(0, 2*np.pi, num_points)
        
        # Parametric equations for Superellipse
        # Exponent p = e2 / 2
        p = e2 / 2.0
        
        # Local coordinates
        x_local = ax * np.sign(np.cos(t)) * (np.abs(np.cos(t)) ** p)
        y_local = ay * np.sign(np.sin(t)) * (np.abs(np.sin(t)) ** p)
        
        # World coordinates
        x_world = cx + x_local
        y_world = cy + y_local
        
        # Transform to Plot coordinates (Rotation)
        x_plot, y_plot = self._transform(x_world, y_world)
        
        return np.column_stack([x_plot, y_plot])

    def update_obstacle(self, center, scale, epsilon):
        """
        Updates the red unsafe region visualization.
        """
        # Remove old patch if exists
        if self.sq_patch is not None:
            self.sq_patch.remove()

        # Generate new points
        verts = self._get_superquadric_contour(center, scale, epsilon)
        
        # Create Polygon patch
        self.sq_patch = Polygon(verts, closed=True, facecolor='red', alpha=0.3, edgecolor='darkred', linestyle='--')
        self.ax.add_patch(self.sq_patch)

    def update_state(self, eef_pos, raw_vel, safe_vel, velocity_scale=1.0):
        """
        Update the robot state and vectors.
        
        Args:
            eef_pos: np.array [x, y, z]
            raw_vel: np.array [vx, vy, vz, ...] (Raw User Input)
            safe_vel: np.array [vx, vy, vz, ...] (CBF Output)
            velocity_scale: Multiplier to make small velocities visible on plot
        """
        # 1. Transform Position
        x, y = self._transform(eef_pos[0], eef_pos[1])
        
        # 2. Transform Vectors
        vx_raw, vy_raw = self._transform_vector(raw_vel[0], raw_vel[1])
        vx_safe, vy_safe = self._transform_vector(safe_vel[0], safe_vel[1])

        # Update EEF Dot
        self.eef_dot.set_data([x], [y])
        
        # Update Vectors (Origins at transformed EEF position)
        # Raw Command
        self.quiver_raw.set_offsets([x, y])
        self.quiver_raw.set_UVC(vx_raw * velocity_scale, vy_raw * velocity_scale)
        
        # Safe Command
        self.quiver_safe.set_offsets([x, y])
        self.quiver_safe.set_UVC(vx_safe * velocity_scale, vy_safe * velocity_scale)

        # Draw frame
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()