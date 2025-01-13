import numpy as np
import tensorflow as tf

from src.mesh.mesh import Mesh
from src.models.model import PINN
from src.training.loss import NavierStokesLoss

class LidDrivenCavity():
    
    def __init__(self, xRange, yRange):
        self.is2D = False

        self.problemTag = "LidDrivenCavity"
        self.mesh  = Mesh(self.is2D)
        self.model = PINN(input_shape=2, output_shape=3)

        self.loss = None
        
        self.xRange = xRange
        self.yRange = yRange

        return
    
    def generateMesh(self, Nx=200, Ny=200, sampling_method='random'):
        # Generate the mesh
        self.mesh.generateMesh(
            x_range=self.xRange,
            y_range=self.yRange,
            Nx=Nx,
            Ny=Ny,
            sampling_method=sampling_method
        )

        # Initialize boundaries
        self.mesh.boundaries = {
            'left': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'right': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'bottom': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'top': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None}
        }

        def setBoundary(boundary_name, x_values, y_values, u_values=None, v_values=None, p_values=None):
            if u_values is not None:
                self.mesh.setBoundaryCodition(x_values, y_values, u_values, 'u', boundary_name)
            if v_values is not None:
                self.mesh.setBoundaryCodition(x_values, y_values, v_values, 'v', boundary_name)
            if p_values is not None:
                self.mesh.setBoundaryCodition(x_values, y_values, p_values, 'p', boundary_name)
            return

        # Define boundary data
        NBoundary = 100
        setBoundary('top',
                    np.linspace(self.xRange[0], self.xRange[1], NBoundary),
                    np.full((NBoundary,), self.yRange[1], dtype=np.float32),
                    np.ones(NBoundary))

        setBoundary('bottom',
                    np.linspace(self.xRange[0], self.xRange[1], NBoundary),
                    np.full((NBoundary,), self.yRange[0], dtype=np.float32),
                    np.zeros(NBoundary), np.zeros(NBoundary))

        setBoundary('left',
                    np.full((NBoundary,), self.xRange[0], dtype=np.float32),
                    np.linspace(self.yRange[0], self.yRange[1], NBoundary),
                    np.zeros(NBoundary), np.zeros(NBoundary))

        setBoundary('right',
                    np.full((NBoundary,), self.xRange[1], dtype=np.float32),
                    np.linspace(self.yRange[0], self.yRange[1], NBoundary),
                    np.zeros(NBoundary), np.zeros(NBoundary))
        
        self.getLossFunction()

        return
    
    def getLossFunction(self):
        self.loss = NavierStokesLoss(self.mesh, self.model)
        self.loss.loss_function()
    
    def train(self):
        self.model.train(self.loss.loss, )