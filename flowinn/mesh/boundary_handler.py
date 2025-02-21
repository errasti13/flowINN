import numpy as np
from typing import Optional, Dict

class BoundaryConditionHandler:
    @staticmethod
    def setBoundaryCondition(mesh: 'Mesh', xCoord: np.ndarray, yCoord: np.ndarray, value: np.ndarray, varName: str,
                             boundaryName: str, zCoord: Optional[np.ndarray] = None, interior: bool = False,
                             bc_type: Optional[str] = None) -> None:
        """
        Sets boundary conditions for either exterior or interior boundaries.

        Args:
            xCoord (np.ndarray): x-coordinates of the boundary.
            yCoord (np.ndarray): y-coordinates of the boundary.
            value (np.ndarray): Value of the boundary condition.
            varName (str): Name of the variable.
            boundaryName (str): Name of the boundary.
            zCoord (Optional[np.ndarray]): z-coordinates of the boundary (for 3D meshes).
            interior (bool): Flag indicating if this is an interior boundary. Defaults to False.
            bc_type (Optional[str]): Type of the boundary condition.
        """
        boundary_dict = mesh._interiorBoundaries if interior else mesh._boundaries

        if boundaryName not in boundary_dict:
            boundary_dict[boundaryName] = {}

        boundary_dict[boundaryName]['x'] = np.asarray(xCoord, dtype=np.float32)
        boundary_dict[boundaryName]['y'] = np.asarray(yCoord, dtype=np.float32)

        if not mesh.is2D:
            if zCoord is None:
                raise ValueError(f"z coordinate required for 3D mesh in boundary {boundaryName}")
            boundary_dict[boundaryName]['z'] = np.asarray(zCoord, dtype=np.float32)

        if value is not None:
            boundary_dict[boundaryName][varName] = np.asarray(value, dtype=np.float32)
            boundary_dict[boundaryName][f'{varName}_type'] = bc_type

        boundary_dict[boundaryName]['isInterior'] = interior

    @staticmethod
    def setPeriodicBoundary(mesh: 'Mesh', boundary_name: str, coupled_boundary: str) -> None:
        """
        Sets a periodic boundary pair.

        Args:
            mesh (Mesh): The mesh object.
            boundary_name (str): Name of the periodic boundary.
            coupled_boundary (str): Name of the boundary to which this boundary is coupled.
        """
        # Check if boundaries exist
        if boundary_name not in mesh._boundaries:
            raise ValueError(f"Boundary {boundary_name} not found in boundaries")
        if coupled_boundary not in mesh._boundaries:
            raise ValueError(f"Coupled boundary {coupled_boundary} not found in boundaries")

        # Create or get the periodic boundary dictionary
        if boundary_name not in mesh._periodicBoundaries:
            mesh._periodicBoundaries[boundary_name] = {}

        # Copy coordinates from the boundary
        boundary_data = mesh._boundaries[boundary_name]
        mesh._periodicBoundaries[boundary_name].update({
            'x': boundary_data['x'],
            'y': boundary_data['y'],
            'z': boundary_data['z'],
            'coupled_boundary': coupled_boundary
        })

