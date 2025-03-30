import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from typing import Optional, List, Dict, Tuple, Union, Any
import os

from flowinn.plot.postprocess import Postprocess
from flowinn.mesh.mesh import Mesh

class Plot:
    """
    An advanced plotting class for visualizing flow solutions and mesh data.
    
    Features:
    - 2D and 3D visualization
    - Static and animated plots
    - Multiple plot types: contour, scatter, streamlines, vector fields
    - Interactive plots with plotly
    - Export capabilities in various formats
    - Customizable plot styling
    
    Attributes:
        mesh (Mesh): The mesh object to plot on
        postprocessor (Postprocess): Optional postprocessor for computing derived quantities
        style (dict): Plot styling configuration
    """
    
    def __init__(self, mesh: Mesh) -> None:
        """
        Initialize Plot object with mesh and default styling.
        
        Args:
            mesh (Mesh): The mesh object to plot on
        """
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a Mesh instance")
            
        self._mesh: Mesh = mesh
        self._postprocessor: Optional[Postprocess] = None
        
        # Default plot styling
        self._style = {
            'figsize': (12, 8),
            'dpi': 300,
            'cmap': 'viridis',
            'scatter_size': 15,
            'scatter_alpha': 0.7,
            'contour_levels': 50,
            'streamline_density': 2,
            'streamline_color': 'white',
            'streamline_width': 1,
            'grid_alpha': 0.3,
            'font_size': 12,
            'title_size': 14,
            'colorbar_label_size': 10,
            'axis_label_size': 11
        }

    @property
    def style(self) -> dict:
        """Get current plot styling configuration."""
        return self._style

    def set_style(self, **kwargs) -> None:
        """
        Update plot styling configuration.
        
        Args:
            **kwargs: Style parameters to update
        """
        self._style.update(kwargs)

    def plot_contour(self, solkey: str, ax: Optional[plt.Axes] = None, 
                    streamlines: bool = False, **kwargs) -> plt.Axes:
        """
        Create a contour plot of the solution field.
        
        Args:
            solkey (str): Solution field to plot
            ax (Optional[plt.Axes]): Matplotlib axes to plot on
            streamlines (bool): Whether to overlay streamlines
            **kwargs: Additional styling parameters
        
        Returns:
            plt.Axes: The matplotlib axes object
        """
        if solkey not in self.mesh.solutions:
            raise KeyError(f"Solution key '{solkey}' not found")
            
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style['figsize'], dpi=self.style['dpi'])

        # For 3D meshes, we'll plot at the middle z-plane
        if not self.mesh.is2D and self.mesh.z is not None:
            # Get unique z values and find the middle plane index
            z_unique = np.unique(self.mesh.z)
            mid_z_idx = len(z_unique) // 2
            mid_z = z_unique[mid_z_idx]
            
            # Filter points in the middle z-plane (with some tolerance)
            tol = 1e-6
            z_mask = np.abs(self.mesh.z.flatten() - mid_z) < tol
            x_flat = self.mesh.x.flatten()[z_mask]
            y_flat = self.mesh.y.flatten()[z_mask]
            sol_flat = self.mesh.solutions[solkey].flatten()[z_mask]
        else:
            x_flat = self.mesh.x.flatten()
            y_flat = self.mesh.y.flatten()
            sol_flat = self.mesh.solutions[solkey].flatten()
            
        # Create uniform grid for plotting
        x_min, x_max = x_flat.min(), x_flat.max()
        y_min, y_max = y_flat.min(), y_flat.max()
        
        # Create perfectly uniform grid for both contour and streamplot
        nx, ny = 200, 200  # Increased resolution
        x_uniform = np.linspace(x_min, x_max, nx)
        y_uniform = np.linspace(y_min, y_max, ny)
        grid_x, grid_y = np.meshgrid(x_uniform, y_uniform)
        
        # Interpolate solution onto uniform grid
        grid_sol = griddata(
            (x_flat, y_flat), 
            sol_flat, 
            (grid_x, grid_y), 
            method='cubic',
            fill_value=np.nan
        )
        
        # Create contour plot
        levels = kwargs.get('levels', self.style['contour_levels'])
        cmap = kwargs.get('cmap', self.style['cmap'])
        cp = ax.contourf(grid_x, grid_y, grid_sol, levels=levels, cmap=cmap)
        
        # Add streamlines if requested
        if streamlines:
            if 'u' not in self.mesh.solutions or 'v' not in self.mesh.solutions:
                raise KeyError("Streamlines require 'u' and 'v' velocity components")
                
            # Get velocity components for the same points
            if not self.mesh.is2D and self.mesh.z is not None:
                u_flat = self.mesh.solutions['u'].flatten()[z_mask]
                v_flat = self.mesh.solutions['v'].flatten()[z_mask]
            else:
                u_flat = self.mesh.solutions['u'].flatten()
                v_flat = self.mesh.solutions['v'].flatten()
                
            # Interpolate velocity components onto the same uniform grid
            u = griddata(
                (x_flat, y_flat),
                u_flat,
                (grid_x, grid_y),
                method='cubic',
                fill_value=0.0
            )
            v = griddata(
                (x_flat, y_flat),
                v_flat,
                (grid_x, grid_y),
                method='cubic',
                fill_value=0.0
            )
            
            # Mask out regions outside the domain or with invalid values
            mask = ~(np.isnan(grid_sol))
            u = np.where(mask, u, 0.0)
            v = np.where(mask, v, 0.0)
            
            # Add streamlines with proper density and styling
            density = kwargs.get('streamline_density', self.style['streamline_density'])
            color = kwargs.get('streamline_color', self.style['streamline_color'])
            linewidth = kwargs.get('streamline_width', self.style['streamline_width'])
            
            # Create streamlines on the uniform grid
            ax.streamplot(
                x_uniform,  # Use 1D arrays for streamplot
                y_uniform,
                u.T,  # Transpose to match streamplot's expected format
                v.T,
                density=density,
                color=color,
                linewidth=linewidth,
                arrowsize=1.0
            )
        
        # Customize plot
        plt.colorbar(cp, ax=ax, label=solkey)
        title = f'{solkey} Field'
        if not self.mesh.is2D:
            title += f' at z/H=0.5'
        ax.set_title(title, fontsize=self.style['title_size'])
        ax.set_xlabel('X', fontsize=self.style['axis_label_size'])
        ax.set_ylabel('Y', fontsize=self.style['axis_label_size'])
        ax.set_aspect('equal')
        
        return ax

    def plot_vector_field(self, scale: float = 1.0, skip: int = 5, 
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot velocity vector field.
        
        Args:
            scale (float): Scaling factor for vectors
            skip (int): Plot every nth vector to avoid crowding
            ax (Optional[plt.Axes]): Matplotlib axes to plot on
            
        Returns:
            plt.Axes: The matplotlib axes object
        """
        if 'u' not in self.mesh.solutions or 'v' not in self.mesh.solutions:
            raise KeyError("Vector field requires 'u' and 'v' velocity components")
            
        if ax is None:
            fig, ax = plt.subplots(figsize=self.style['figsize'], dpi=self.style['dpi'])
            
        x = self.mesh.x[::skip]
        y = self.mesh.y[::skip]
        u = self.mesh.solutions['u'][::skip]
        v = self.mesh.solutions['v'][::skip]
        
        # Calculate velocity magnitude for color mapping
        magnitude = np.sqrt(u**2 + v**2)
        
        # Plot vectors
        ax.quiver(x, y, u, v, magnitude,
                 scale=scale,
                 cmap=self.style['cmap'],
                 width=0.005)
                 
        plt.colorbar(ax.collections[0], ax=ax, label='Velocity Magnitude')
        ax.set_title('Velocity Field', fontsize=self.style['title_size'])
        ax.set_xlabel('X', fontsize=self.style['axis_label_size'])
        ax.set_ylabel('Y', fontsize=self.style['axis_label_size'])
        ax.set_aspect('equal')
        
        return ax

    def plot_3d_surface(self, solkey: str, elev: float = 30, 
                       azim: float = 45) -> plt.Axes:
        """
        Create a 3D surface plot of the solution field.
        
        Args:
            solkey (str): Solution field to plot
            elev (float): Elevation viewing angle
            azim (float): Azimuth viewing angle
            
        Returns:
            plt.Axes: The matplotlib axes object
        """
        if solkey not in self.mesh.solutions:
            raise KeyError(f"Solution key '{solkey}' not found")
            
        fig = plt.figure(figsize=self.style['figsize'], dpi=self.style['dpi'])
        ax = fig.add_subplot(111, projection='3d')
        
        x, y = self.mesh.x, self.mesh.y
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )
        
        sol = self.mesh.solutions[solkey]
        grid_sol = griddata((x.flatten(), y.flatten()), sol.flatten(),
                          (grid_x, grid_y), method='cubic')
                          
        surf = ax.plot_surface(grid_x, grid_y, grid_sol,
                             cmap=self.style['cmap'],
                             linewidth=0,
                             antialiased=True)
                             
        plt.colorbar(surf, ax=ax, label=solkey)
        ax.set_title(f'3D {solkey} Field', fontsize=self.style['title_size'])
        ax.set_xlabel('X', fontsize=self.style['axis_label_size'])
        ax.set_ylabel('Y', fontsize=self.style['axis_label_size'])
        ax.set_zlabel(solkey, fontsize=self.style['axis_label_size'])
        
        ax.view_init(elev=elev, azim=azim)
        return ax

    def create_animation(self, solkey: str, frames: List[np.ndarray], 
                        interval: int = 50) -> FuncAnimation:
        """
        Create an animation of time-varying solution fields.
        
        Args:
            solkey (str): Solution field to animate
            frames (List[np.ndarray]): List of solution arrays for each time step
            interval (int): Time between frames in milliseconds
            
        Returns:
            FuncAnimation: The animation object
        """
        fig, ax = plt.subplots(figsize=self.style['figsize'], dpi=self.style['dpi'])
        
        x, y = self.mesh.x, self.mesh.y
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )
        
        def update(frame):
            ax.clear()
            grid_sol = griddata((x.flatten(), y.flatten()), frame.flatten(),
                              (grid_x, grid_y), method='cubic')
            cp = ax.contourf(grid_x, grid_y, grid_sol,
                           levels=self.style['contour_levels'],
                           cmap=self.style['cmap'])
            return [cp]
            
        anim = FuncAnimation(fig, update, frames=frames,
                           interval=interval, blit=True)
        return anim

    def plot_interactive(self, solkey: str) -> None:
        """
        Create an interactive plot using plotly.
        
        Args:
            solkey (str): Solution field to plot
        """
        if solkey not in self.mesh.solutions:
            raise KeyError(f"Solution key '{solkey}' not found")
            
        x, y = self.mesh.x, self.mesh.y
        sol = self.mesh.solutions[solkey]
        
        fig = go.Figure(data=[
            go.Scatter(
                x=x.flatten(),
                y=y.flatten(),
                mode='markers',
                marker=dict(
                    size=5,
                    color=sol.flatten(),
                    colorscale='Jet',
                    showscale=True,
                    colorbar=dict(title=solkey)
                )
            )
        ])
        
        fig.update_layout(
            title=f'Interactive {solkey} Field',
            xaxis_title='X',
            yaxis_title='Y',
            width=800,
            height=600
        )
        
        fig.show()

    def export_plot(self, filename: str, solkey: str, **kwargs) -> None:
        """
        Export plot to various file formats.
        
        Args:
            filename (str): Output filename with extension
            solkey (str): Solution field to plot
            **kwargs: Additional plotting parameters
        """
        extension = os.path.splitext(filename)[1].lower()
        
        fig, ax = plt.subplots(figsize=self.style['figsize'], dpi=self.style['dpi'])
        self.plot_contour(solkey, ax=ax, **kwargs)
        
        if extension in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']:
            plt.savefig(filename, bbox_inches='tight', dpi=self.style['dpi'])
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        plt.close(fig)

    def plot_comparison(self, solkeys: List[str], titles: Optional[List[str]] = None,
                       layout: Tuple[int, int] = None) -> None:
        """
        Create a multi-panel comparison plot of different solution fields.
        
        Args:
            solkeys (List[str]): List of solution fields to compare
            titles (Optional[List[str]]): List of subplot titles
            layout (Tuple[int, int]): Number of rows and columns for subplots
        """
        n = len(solkeys)
        if not layout:
            cols = min(3, n)
            rows = (n + cols - 1) // cols
        else:
            rows, cols = layout
            
        fig, axes = plt.subplots(rows, cols,
                                figsize=(cols*6, rows*5),
                                dpi=self.style['dpi'])
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (ax, key) in enumerate(zip(axes, solkeys)):
            self.plot_contour(key, ax=ax)
            if titles:
                ax.set_title(titles[i], fontsize=self.style['title_size'])
                
        # Hide empty subplots
        for ax in axes[len(solkeys):]:
            ax.set_visible(False)
            
        plt.tight_layout()

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @mesh.setter
    def mesh(self, value: Mesh) -> None:
        if not isinstance(value, Mesh):
            raise TypeError("mesh must be a Mesh instance")
        self._mesh = value

    @property
    def postprocessor(self) -> Optional[Postprocess]:
        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, value: Postprocess) -> None:
        if not isinstance(value, Postprocess):
            raise TypeError("postprocessor must be a Postprocess instance")
        self._postprocessor = value

    def plot(self, solkey: str, streamlines: bool) -> None:
        """
        Plots the solution field using contour plots and streamlines (optional).

        Args:
            solkey (str): The key of the solution field to plot.
            streamlines (bool): Whether to plot streamlines.

        Raises:
            KeyError: If the solution key is not found in the mesh solutions or if streamline plotting requires missing velocity components.
        """
        from scipy.interpolate import griddata

        if solkey == 'vMag' and 'vMag' not in self.mesh.solutions:
            if self.postprocessor is None:
                raise ValueError("Postprocessor is required to compute velocity magnitude")
            self.postprocessor.compute_velocity_magnitude()

        if solkey not in self.mesh.solutions:
            raise KeyError(
                f"The solution key '{solkey}' was not found in the available solutions. "
                f"Available keys are: {list(self.mesh.solutions.keys())}."
            )

        x = self.mesh.x
        y = self.mesh.y
        sol = self.mesh.solutions[solkey]

        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))

        grid_sol = griddata((x, y), sol, (grid_x, grid_y), method='cubic')

        plt.figure(figsize=(8, 6))
        plt.title(f'Solution Field {solkey}')

        cp = plt.contourf(grid_x, grid_y, grid_sol, cmap='jet', levels=50)
        plt.colorbar(cp)

        if streamlines:
            if 'u' not in self.mesh.solutions or 'v' not in self.mesh.solutions:
                raise KeyError("Streamline plotting requires 'u' and 'v' velocity components in solutions.")

            u = self.mesh.solutions['u']
            v = self.mesh.solutions['v']
            grid_u = griddata((x, y), u, (grid_x, grid_y), method='cubic')
            grid_v = griddata((x, y), v, (grid_x, grid_y), method='cubic')

            plt.streamplot(grid_x, grid_y, grid_u, grid_v, color='k', linewidth=1)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def scatterPlot(self, solkey: str, title: str = None, savePath: str = None, show: bool = False) -> None:
        """
        Visualizes the solution field using scatter plot with boundaries.

        Args:
            solkey (str): The key of the solution field to plot.
            title (str, optional): Custom title for the plot. If None, uses solkey.
        """
        x = self.mesh.x.flatten()
        y = self.mesh.y.flatten()

        is3D = not self.mesh.is2D and self.mesh.z is not None
        z = self.mesh.z.flatten() if is3D else None

        sol = self.mesh.solutions[solkey]

        if is3D:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(x, y, z,
                               c=sol,
                               s=20,
                               alpha=0.6,
                               cmap='jet')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
            mid_x = (x.max()+x.min()) * 0.5
            mid_y = (y.max()+y.min()) * 0.5
            mid_z = (z.max()+z.min()) * 0.5
            ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
            ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
            ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)

            plt.colorbar(scatter, label=solkey)
            plt.title(title if title else f'Solution Field: {solkey}')

        else:
            sol = self.mesh.solutions[solkey]
            plt.figure(figsize=(10, 8))
            plt.title(title if title else f'Solution Field: {solkey}', fontsize=12)

            plt.set_cmap('viridis')

            scatter = plt.scatter(x, y,
                                c=sol,
                                s=20,
                                alpha=0.6,
                                cmap='jet',
                                zorder=2)

            cbar = plt.colorbar(scatter, label=solkey)
            cbar.ax.tick_params(labelsize=10)

            for boundary_data in self.mesh.boundaries.values():
                x_boundary = boundary_data['x']
                y_boundary = boundary_data['y']
                plt.plot(x_boundary, y_boundary,
                        'k-',
                        linewidth=1.5,
                        zorder=3,
                        label='Exterior Boundary')

            if self.mesh.interiorBoundaries:
                for boundary_data in self.mesh.interiorBoundaries.values():
                    x_boundary = boundary_data['x']
                    y_boundary = boundary_data['y']
                    plt.plot(x_boundary, y_boundary,
                            'r-',
                            linewidth=2,
                            zorder=3,
                            label='Interior Boundary')

            plt.xlabel('X', fontsize=11)
            plt.ylabel('Y', fontsize=11)
            plt.axis('equal')

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(),
                    loc='upper right',
                    framealpha=0.9,
                    fontsize=10)

            plt.grid(True, linestyle='--', alpha=0.3, zorder=1)

            plt.tight_layout()

        plt.tight_layout()
        if savePath is not None:
            plt.savefig(savePath)

        if show:
            plt.show()

        plt.close()

    def plotSlices(self, solkey: str, num_points: int = 50, z_cuts: Optional[list] = None) -> None:
        """
        Create slice plots for 3D solution fields using interpolation onto regular grids.

        Args:
            solkey (str): Solution field to plot.
            num_points (int): Number of points for interpolation grid.
            z_cuts (Optional[list]): List of z-positions for slices (between 0 and 1), default is [0.25, 0.5, 0.75].

        Raises:
            ValueError: If slice plotting is attempted on a 2D mesh.
        """
        if self.mesh.is2D:
            raise ValueError("Slice plotting is only available for 3D meshes")

        from scipy.interpolate import griddata

        x = self.mesh.x.flatten()
        y = self.mesh.y.flatten()
        z = self.mesh.z.flatten()
        sol = self.mesh.solutions[solkey]

        x_unique = np.linspace(x.min(), x.max(), num_points)
        y_unique = np.linspace(y.min(), y.max(), num_points)
        z_unique = np.linspace(z.min(), z.max(), num_points)

        if z_cuts is None:
            z_cuts = [0.25, 0.5, 0.75]

        z_positions = [z.min() + cut * (z.max() - z.min()) for cut in z_cuts]
        n_cuts = len(z_positions)

        fig = plt.figure(figsize=(5*n_cuts, 4))

        for idx, z_pos in enumerate(z_positions):
            ax = fig.add_subplot(1, n_cuts, idx+1)

            xx_xy, yy_xy = np.meshgrid(x_unique, y_unique)
            zz_xy = np.full_like(xx_xy, z_pos)
            points_xy = np.column_stack((xx_xy.flatten(), yy_xy.flatten(), zz_xy.flatten()))

            sol_xy = griddata((x, y, z), sol, points_xy, method='linear')
            sol_xy = sol_xy.reshape(xx_xy.shape)

            im = ax.contourf(xx_xy, yy_xy, sol_xy, levels=50, cmap='jet')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'{solkey} at z/H={z_cuts[idx]:.2f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        x_mid = x_unique[num_points//2]
        yy_yz, zz_yz = np.meshgrid(y_unique, z_unique)
        xx_yz = np.full_like(yy_yz, x_mid)
        points_yz = np.column_stack((xx_yz.flatten(), yy_yz.flatten(), zz_yz.flatten()))
        sol_yz = griddata((x, y, z), sol, points_yz, method='linear')
        sol_yz = sol_yz.reshape(yy_yz.shape)

        im1 = ax1.contourf(yy_yz, zz_yz, sol_yz, levels=50, cmap='jet')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title(f'{solkey} at x/L=0.5')
        ax1.set_xlabel('Y')
        ax1.set_ylabel('Z')
        ax1.set_aspect('equal')

        y_mid = y_unique[num_points//2]
        xx_xz, zz_xz = np.meshgrid(x_unique, z_unique)
        yy_xz = np.full_like(xx_xz, y_mid)
        points_xz = np.column_stack((xx_xz.flatten(), yy_xz.flatten(), zz_xz.flatten()))
        sol_xz = griddata((x, y, z), sol, points_xz, method='linear')
        sol_xz = sol_xz.reshape(xx_xz.shape)

        im2 = ax2.contourf(xx_xz, zz_xz, sol_xz, levels=50, cmap='jet')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title(f'{solkey} at y/H=0.5')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_aspect('equal')

        plt.tight_layout()
        plt.show()

    def vectorField(self, xRange=None, yRange=None, scale=50, density=2):
        """
        Create a vector field plot using quiver.
        
        Args:
            xRange (tuple): Domain x-range (min_x, max_x)
            yRange (tuple): Domain y-range (min_y, max_y)
            scale (float): Scaling factor for arrows (default: 50)
            density (int): Arrow density (default: 2)
        """
        if not all(key in self.mesh.solutions for key in ['u', 'v']):
            raise ValueError("Velocity components 'u' and 'v' are required for vector field plots")
        
        # Use provided ranges or fall back to mesh extents
        xMin = xRange[0] if xRange is not None else self.mesh.x.min()
        xMax = xRange[1] if xRange is not None else self.mesh.x.max()
        yMin = yRange[0] if yRange is not None else self.mesh.y.min()
        yMax = yRange[1] if yRange is not None else self.mesh.y.max()
        
        # Create a regular grid for interpolation
        nx, ny = 50, 50  # Number of points in each direction
        x = np.linspace(xMin, xMax, nx)
        y = np.linspace(yMin, yMax, ny)
        X, Y = np.meshgrid(x, y)
        
        # Interpolate velocity components onto regular grid
        from scipy.interpolate import griddata
        points = np.column_stack((self.mesh.x.flatten(), self.mesh.y.flatten()))
        U = griddata(points, self.mesh.solutions['u'], (X, Y), method='cubic')
        V = griddata(points, self.mesh.solutions['v'], (X, Y), method='cubic')
        
        skip = (slice(None, None, density), slice(None, None, density))
        
        plt.figure(figsize=self.style['figsize'])
        plt.quiver(X[skip], Y[skip], U[skip], V[skip], 
                  np.sqrt(U[skip]**2 + V[skip]**2),
                  scale=scale, cmap=self.style['cmap'])
        plt.colorbar(label='Velocity Magnitude')
        
        # Plot boundaries if they exist
        self._plotBoundaries()
        
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Flow Field')
        plt.show()

    def _plotBoundaries(self):
        """Plot exterior and interior boundaries if they exist."""
        # Plot exterior boundaries
        if hasattr(self.mesh, 'boundaries'):
            for boundary_data in self.mesh.boundaries.values():
                if 'x' in boundary_data and 'y' in boundary_data:
                    plt.plot(boundary_data['x'], boundary_data['y'],
                            'k-', linewidth=1.5, zorder=3)
        
        # Plot interior boundaries (e.g., airfoil)
        if hasattr(self.mesh, 'interiorBoundaries'):
            for boundary_data in self.mesh.interiorBoundaries.values():
                if 'x' in boundary_data and 'y' in boundary_data:
                    plt.plot(boundary_data['x'], boundary_data['y'],
                            'r-', linewidth=2, zorder=3)

