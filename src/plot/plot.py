import matplotlib.pyplot as plt
import numpy as np
from src.plot.postprocess import Postprocess
from src.mesh.mesh import Mesh  # Import Mesh class
from typing import Optional, Tuple, Dict


class Plot:
    """
    A class for plotting solution fields on a given mesh.

    Attributes:
        mesh (Mesh): The mesh object to plot on.
        postprocessor (Postprocess): An optional postprocessor for computing derived quantities.
    """

    def __init__(self, mesh: Mesh) -> None:
        """
        Initializes a new Plot object.

        Args:
            mesh (Mesh): The mesh object to plot on.
        """
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a Mesh instance")

        self._mesh: Mesh = mesh
        self._postprocessor: Optional[Postprocess] = None

    @property
    def mesh(self) -> Mesh:
        """
        Returns the mesh object.
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value: Mesh) -> None:
        """
        Sets the mesh object.

        Args:
            value (Mesh): The new mesh object.

        Raises:
            TypeError: If value is not a Mesh instance.
        """
        if not isinstance(value, Mesh):
            raise TypeError("mesh must be a Mesh instance")
        self._mesh = value

    @property
    def postprocessor(self) -> Optional[Postprocess]:
        """
        Returns the postprocessor object.
        """
        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, value: Postprocess) -> None:
        """
        Sets the postprocessor object.

        Args:
            value (Postprocess): The new postprocessor object.

        Raises:
            TypeError: If value is not a Postprocess instance.
        """
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

    def scatterPlot(self, solkey: str) -> None:
        """
        Visualizes the solution field using scatter plot with boundaries.

        Args:
            solkey (str): The key of the solution field to plot.
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
            plt.title(f'Solution Field: {solkey}')

        else:
            sol = self.mesh.solutions[solkey]
            plt.figure(figsize=(10, 8))
            plt.title(f'Solution Field: {solkey}', fontsize=12)

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
            plt.show()

        plt.tight_layout()
        plt.show()

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

