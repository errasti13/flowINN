import numpy as np
import matplotlib.pyplot as plt

from src.mesh.mesh import Mesh
from src.models.model import PINN
from src.training.loss import NavierStokesLoss

class LidDrivenCavity():
    
    def __init__(self, caseName, xRange, yRange):
        self.is2D = False

        self.problemTag = caseName
        self.mesh  = Mesh(self.is2D)
        self.model = PINN(input_shape=2, output_shape=3, eq = self.problemTag)

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
                    u_values=np.ones(NBoundary))

        setBoundary('bottom',
                    np.linspace(self.xRange[0], self.xRange[1], NBoundary),
                    np.full((NBoundary,), self.yRange[0], dtype=np.float32),
                    u_values = np.zeros(NBoundary), v_values = np.zeros(NBoundary))

        setBoundary('left',
                    np.full((NBoundary,), self.xRange[0], dtype=np.float32),
                    np.linspace(self.yRange[0], self.yRange[1], NBoundary),
                    u_values = np.zeros(NBoundary), v_values = np.zeros(NBoundary))

        setBoundary('right',
                    np.full((NBoundary,), self.xRange[1], dtype=np.float32),
                    np.linspace(self.yRange[0], self.yRange[1], NBoundary),
                    u_values = np.zeros(NBoundary), v_values = np.zeros(NBoundary))
        return
    
    def getLossFunction(self):
        self.loss = NavierStokesLoss(self.mesh, self.model)
    
    def train(self, epochs=10000, print_interval=100,  autosaveInterval=10000):
        self.getLossFunction()
        self.model.train(self.loss.loss_function, epochs=epochs, print_interval=print_interval,autosave_interval=autosaveInterval)

    def predict(self):
        X = (np.hstack((self.mesh.X.flatten()[:, None], self.mesh.Y.flatten()[:, None])))
        sol = self.model.predict(X)

        self.mesh.solutions['u'] = sol[:, 0]
        self.mesh.solutions['v'] = sol[:, 1]
        self.mesh.solutions['p'] = sol[:, 2]

        return

    def plot(self):
        plt.figure()
        plt.scatter(self.mesh.X.flatten(), self.mesh.Y.flatten(), 
                    c=self.mesh.solutions['u'], 
                    cmap='viridis')  # Add a colormap
        plt.colorbar()  # Add a colorbar to show the scale
        plt.show()

        plt.figure()
        plt.scatter(self.mesh.X.flatten(), self.mesh.Y.flatten(), 
                    c=self.mesh.solutions['v'], 
                    cmap='viridis')  # Add a colormap
        plt.colorbar()  # Add a colorbar to show the scale
        plt.show()