import numpy as np
from typing import Optional, List, Tuple
from scipy.spatial import Delaunay
from .geometry import GeometryUtils

class Sampler:
    @staticmethod
    def _sampleRandomlyWithinBoundary(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                                     z_boundary: Optional[np.ndarray], Nx: int, Ny: int,
                                     Nz: Optional[int]) -> None:
        """
        Samples points randomly within the defined boundary.

        Args:
            x_boundary (np.ndarray): x-coordinates of the boundary.
            y_boundary (np.ndarray): y-coordinates of the boundary.
            z_boundary (Optional[np.ndarray]): z-coordinates of the boundary (for 3D meshes).
            Nx (int): Number of points in the x-dimension.
            Ny (int): Number of points in the y-dimension.
            Nz (Optional[int]): Number of points in the z-dimension.

        Raises:
            ValueError: If boundary coordinates contain NaN values.
        """
        if mesh.is2D:
            Sampler._sampleRandom2D(mesh, x_boundary, y_boundary, Nx, Ny)
        else:
            Sampler._sampleRandom3D(mesh, x_boundary, y_boundary, z_boundary, Nx, Ny, Nz)


    @staticmethod
    def _sampleRandom2D(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                         Nx: int, Ny: int) -> None:
        
        try:
            x_boundary = np.asarray(x_boundary, dtype=np.float32)
            y_boundary = np.asarray(y_boundary, dtype=np.float32)

            if np.any(np.isnan(x_boundary)) or np.any(np.isnan(y_boundary)):
                raise ValueError("Boundary coordinates contain NaN values")

            Nt = Nx * Ny

            samples: List[np.ndarray] = []
            while len(samples) < Nt:
                x_rand = np.random.uniform(np.min(x_boundary), np.max(x_boundary), size=Nt)
                y_rand = np.random.uniform(np.min(y_boundary), np.max(y_boundary), size=Nt)

                points = np.column_stack((x_rand, y_rand))

                valid_points = Sampler._check_points_in_domain(mesh, points)
                samples.extend(valid_points)

            samples = np.array(samples)[:Nt]

            mesh._x = samples[:, 0].reshape(Nx, Ny)
            mesh._y = samples[:, 1].reshape(Nx, Ny)

        except Exception as e:
            print(f"Debug: Error during random sampling: {str(e)}")
            raise
        
    @staticmethod
    def _sampleRandom3D(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                                      z_boundary: np.ndarray, Nx: int, Ny: int,
                                      Nz: int) -> None:
        try:
            x_boundary = np.asarray(x_boundary, dtype=np.float32)
            y_boundary = np.asarray(y_boundary, dtype=np.float32)
            z_boundary = np.asarray(z_boundary, dtype=np.float32)

            if np.any(np.isnan(x_boundary)) or np.any(np.isnan(y_boundary)) or \
                (np.any(np.isnan(z_boundary))):
                raise ValueError("Boundary coordinates contain NaN values")

            Nt = Nx * Ny * Nz

            samples: List[np.ndarray] = []
            while len(samples) < Nt:
                x_rand = np.random.uniform(np.min(x_boundary), np.max(x_boundary), size=Nt)
                y_rand = np.random.uniform(np.min(y_boundary), np.max(y_boundary), size=Nt)
                z_rand = np.random.uniform(np.min(z_boundary), np.max(z_boundary), size=Nt)
                points = np.column_stack((x_rand, y_rand, z_rand))

                samples.extend(points)

            samples = np.array(samples)[:Nt]

            mesh._x = samples[:, 0].reshape(Nx, Ny, Nz)
            mesh._y = samples[:, 1].reshape(Nx, Ny, Nz)
            mesh._z = samples[:, 2].reshape(Nx, Ny, Nz)

        except Exception as e:
            print(f"Debug: Error during random sampling: {str(e)}")
            raise

    @staticmethod
    def _sampleUniformlyWithinBoundary(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                                      z_boundary: Optional[np.ndarray], Nx: int, Ny: int,
                                      Nz: Optional[int]) -> None:
        """
        Samples points uniformly within the defined boundary.

        Args:
            x_boundary (np.ndarray): x-coordinates of the boundary.
            y_boundary (np.ndarray): y-coordinates of the boundary.
            z_boundary (Optional[np.ndarray]): z-coordinates of the boundary (for 3D meshes).
            Nx (int): Number of points in the x-dimension.
            Ny (int): Number of points in the y-dimension.
            Nz (Optional[int]): Number of points in the z-dimension.
        """
        if mesh.is2D:
            Sampler._sampleUniform2D(mesh, x_boundary, y_boundary, Nx, Ny)
        else:
            Sampler._sampleUniform3D(mesh, x_boundary, y_boundary, z_boundary, Nx, Ny, Nz)

    @staticmethod
    def _sampleUniform2D(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                                    Nx: int, Ny: int) -> None:
        
        x_min, x_max = np.min(x_boundary), np.max(x_boundary)
        y_min, y_max = np.min(y_boundary), np.max(y_boundary)
        
        # Create a more dense grid to ensure we have enough points after filtering
        density_factor = 2
        dense_Nx = Nx * density_factor
        dense_Ny = Ny * density_factor
        
        # Create uniform grid
        x_grid, y_grid = np.meshgrid(
            np.linspace(x_min, x_max, dense_Nx),
            np.linspace(y_min, y_max, dense_Ny)
        )
        
        # Stack coordinates for testing
        grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
        
        # Use point-in-polygon test to filter points
        valid_points = Sampler._check_points_in_domain(mesh, grid_points)
        
        # If we have too few points, increase density and try again
        while len(valid_points) < Nx * Ny:
            density_factor *= 2
            dense_Nx = Nx * density_factor
            dense_Ny = Ny * density_factor
            
            x_grid, y_grid = np.meshgrid(
                np.linspace(x_min, x_max, dense_Nx),
                np.linspace(y_min, y_max, dense_Ny)
            )
            grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
            valid_points = Sampler._check_points_in_domain(mesh, grid_points)
        
        # Select evenly spaced points from valid points
        if len(valid_points) > Nx * Ny:
            indices = np.linspace(0, len(valid_points)-1, Nx * Ny, dtype=int)
            valid_points = valid_points[indices]
        
        mesh._x = valid_points[:, 0].reshape(Nx, Ny).astype(np.float32)
        mesh._y = valid_points[:, 1].reshape(Nx, Ny).astype(np.float32)
    
        
    @staticmethod
    def _sampleUniform3D(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                                      z_boundary: np.ndarray, Nx: int, Ny: int,
                                      Nz: int) -> None:
        """
        Samples points uniformly within the defined boundary.

        Args:
            x_boundary (np.ndarray): x-coordinates of the boundary.
            y_boundary (np.ndarray): y-coordinates of the boundary.
            z_boundary (Optional[np.ndarray]): z-coordinates of the boundary (for 3D meshes).
            Nx (int): Number of points in the x-dimension.
            Ny (int): Number of points in the y-dimension.
            Nz (Optional[int]): Number of points in the z-dimension.
        """
        x_min, x_max = np.min(x_boundary), np.max(x_boundary)
        y_min, y_max = np.min(y_boundary), np.max(y_boundary)
        z_min, z_max = (np.min(z_boundary), np.max(z_boundary)) if z_boundary is not None else (None, None)

        x_grid, y_grid, z_grid = np.meshgrid(
            np.linspace(x_min, x_max, Nx),
            np.linspace(y_min, y_max, Ny),
            np.linspace(z_min, z_max, Nz) if z_boundary is not None else [0]
        )
        grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten())) if not mesh.is2D else np.column_stack((x_grid.flatten(), y_grid.flatten()))

        points = np.column_stack((x_boundary, y_boundary, z_boundary)) if not mesh.is2D else np.column_stack((x_boundary, y_boundary))
        triangulation = Delaunay(points)

        inside = triangulation.find_simplex(grid_points) >= 0
        inside_points = grid_points[inside]

        if mesh._interiorBoundaries:
            for boundary_data in mesh._interiorBoundaries.values():
                x_int = boundary_data['x'].flatten()
                y_int = boundary_data['y'].flatten()
                if not mesh.is2D:
                    z_int = boundary_data['z'].flatten()
                    interior_points = np.column_stack((x_int, y_int, z_int))
                else:
                    interior_points = np.column_stack((x_int, y_int))

                interior_tri = Delaunay(interior_points)

                inside_interior = interior_tri.find_simplex(inside_points) >= 0

                inside_points = inside_points[~inside_interior]

        mesh._x, mesh._y, mesh._z = inside_points[:, 0].astype(np.float32), inside_points[:, 1].astype(np.float32), inside_points[:, 2].astype(np.float32)

    @staticmethod
    def _check_points_in_domain(mesh: 'Mesh', points: np.ndarray) -> np.ndarray:
        """Checks if points are inside the domain and outside interior boundaries."""
        return GeometryUtils.check_points_in_domain(points, mesh.boundaries, mesh._interiorBoundaries)