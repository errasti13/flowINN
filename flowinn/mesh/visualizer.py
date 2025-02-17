import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

class MeshVisualizer:
    @staticmethod
    def showMesh(mesh: 'Mesh', figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Visualizes the mesh with proper dimensional scaling.

        Args:
            mesh (Mesh): The mesh to visualize
            figsize (Tuple[int, int]): Size of the figure. Defaults to (8, 6).

        Raises:
            ValueError: If the mesh has not been generated yet.
        """
        if mesh._x is None or mesh._y is None:
            raise ValueError("Mesh has not been generated yet")

        x_min, x_max = np.min(mesh._x), np.max(mesh._x)
        y_min, y_max = np.min(mesh._y), np.max(mesh._y)

        x_min, x_max = np.min(mesh.x), np.max(mesh.x)
        y_min, y_max = np.min(mesh.y), np.max(mesh.y)

        if not mesh.is2D and mesh.z is not None:
            z_min, z_max = np.min(mesh.z), np.max(mesh.z)
            domain_size = f"L×H×W: {x_max-x_min:.1f}×{y_max-y_min:.1f}×{z_max-z_min:.1f}"

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            max_range = max(x_range, y_range, z_range)

            x_mid = (x_max + x_min) / 2
            y_mid = (y_max + y_min) / 2
            z_mid = (z_max + z_min) / 2

            ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
            ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
            ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

            scatter = ax.scatter(mesh.x, mesh.y, mesh.z,
                            c='black', s=1, alpha=0.5)

            for boundary_data in mesh.boundaries.values():
                x_boundary = boundary_data['x']
                y_boundary = boundary_data['y']
                z_boundary = boundary_data['z']
                ax.plot3D(x_boundary, y_boundary, z_boundary,
                        'b-', linewidth=1, alpha=0.5)

            ax.set_xlabel(f'X')
            ax.set_ylabel(f'Y')
            ax.set_zlabel(f'Z')

            ax.set_box_aspect([x_range/max_range,
                            y_range/max_range,
                            z_range/max_range])

        else:
            plt.figure(figsize=figsize)
            domain_size = f"L×H: {x_max-x_min:.1f}×{y_max-y_min:.1f}"

            plt.scatter(mesh.x, mesh.y, c='black', s=1, alpha=0.5)

            for boundary_data in mesh.boundaries.values():
                plt.plot(boundary_data['x'], boundary_data['y'],
                        'b-', linewidth=1, alpha=0.5)

            plt.xlabel(f'X ({x_max-x_min:.1f})')
            plt.ylabel(f'Y ({y_max-y_min:.1f})')
            plt.axis('equal')

        plt.title(f'Mesh Visualization\n{domain_size}')
        plt.tight_layout()
        plt.show()