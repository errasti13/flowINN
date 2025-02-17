import numpy as np
from typing import Dict, Optional

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
