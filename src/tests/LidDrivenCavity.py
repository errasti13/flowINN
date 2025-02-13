import numpy as np
from src.mesh.mesh import Mesh
from src.models.model import PINN
from src.training.loss import NavierStokesLoss
from src.plot.plot import Plot
from src.physics.boundary_conditions import MovingWallBC, WallBC

class LidDrivenCavity():
    
    def __init__(self, caseName, xRange, yRange):
        self.is2D = True

        self.problemTag = caseName
        self.mesh  = Mesh(self.is2D)
        self.model = PINN(input_shape=(2,), output_shape=3, eq = self.problemTag, layers=[20,40,60,40,20])

        self.loss = None
        self.Plot = None
        
        self.xRange = xRange
        self.yRange = yRange

        # Initialize boundary conditions
        self.moving_wall_bc = MovingWallBC("top_wall")
        self.wall_bc = WallBC("wall")

        return
    
    def generateMesh(self, Nx=100, Ny=100, NBoundary=100, sampling_method='random'):
        # Initialize boundaries with new BC system
        self.mesh.boundaries = {
            'left': {
                'x': np.full((NBoundary, 1), self.xRange[0], dtype=np.float32),
                'y': np.linspace(self.yRange[0], self.yRange[1], NBoundary),
                'conditions': {
                    'u': {'value': 0.0},
                    'v': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'x'}
                },
                'bc_type': self.wall_bc
            },
            'right': {
                'x': np.full((NBoundary, 1), self.xRange[1], dtype=np.float32),
                'y': np.linspace(self.yRange[0], self.yRange[1], NBoundary),
                'conditions': {
                    'u': {'value': 0.0},
                    'v': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'x'}
                },
                'bc_type': self.wall_bc
            },
            'bottom': {
                'x': np.linspace(self.xRange[0], self.xRange[1], NBoundary),
                'y': np.full((NBoundary, 1), self.yRange[0], dtype=np.float32),
                'conditions': {
                    'u': {'value': 0.0},
                    'v': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'y'}
                },
                'bc_type': self.wall_bc
            },
            'top': {
                'x': np.linspace(self.xRange[0], self.xRange[1], NBoundary),
                'y': np.full((NBoundary, 1), self.yRange[1], dtype=np.float32),
                'conditions': {
                    'u': {'value': 1.0},  # Moving wall
                    'v': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'y'}
                },
                'bc_type': self.moving_wall_bc
            }
        }
        
        # Set boundary coordinates
        self.mesh.setBoundary('top',
                    np.linspace(self.xRange[0], self.xRange[1], NBoundary),
                    np.full((NBoundary, 1), self.yRange[1], dtype=np.float32))

        self.mesh.setBoundary('bottom',
                    np.linspace(self.xRange[0], self.xRange[1], NBoundary),
                    np.full((NBoundary, 1), self.yRange[0], dtype=np.float32))

        self.mesh.setBoundary('left',
                    np.full((NBoundary, 1), self.xRange[0], dtype=np.float32),
                    np.linspace(self.yRange[0], self.yRange[1], NBoundary))

        self.mesh.setBoundary('right',
                    np.full((NBoundary, 1), self.xRange[1], dtype=np.float32),
                    np.linspace(self.yRange[0], self.yRange[1], NBoundary))
        
        # Generate the mesh
        self.mesh.generateMesh(
            Nx=Nx,
            Ny=Ny,
            sampling_method=sampling_method
        )
        return
    
    def getLossFunction(self):
        self.loss = NavierStokesLoss(self.mesh, self.model)
    
    def train(self, epochs=10000, print_interval=100, autosaveInterval=10000, num_batches=10):
        """Train the model with batch support."""
        self.getLossFunction()
        self.model.train(
            self.loss.loss_function,
            self.mesh,
            epochs=epochs,
            print_interval=print_interval,
            autosave_interval=autosaveInterval,
            num_batches=num_batches
        )

    def predict(self):
        X = (np.hstack((self.mesh.x.flatten()[:, None], self.mesh.y.flatten()[:, None])))
        sol = self.model.predict(X)

        self.mesh.solutions['u'] = sol[:, 0]
        self.mesh.solutions['v'] = sol[:, 1]
        self.mesh.solutions['p'] = sol[:, 2]

        self.generate_plots()  # Generate plots after prediction

        return
    
    def generate_plots(self):
        self.Plot = Plot(self.mesh)

    def plot(self, solkey = 'u', streamlines = False):
        self.Plot.plot(solkey, streamlines)

    def load_model(self):
        self.model.load(self.problemTag)
