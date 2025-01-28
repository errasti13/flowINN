import numpy as np

from src.mesh.mesh import Mesh
from src.models.model import PINN
from src.training.loss import NavierStokesLoss
from src.plot.plot import Plot

class FlowOverAirfoil():
    
    def __init__(self, caseName, xRange, yRange, AoA = 0.0):
        self.is2D = True

        self.problemTag = caseName
        self.mesh  = Mesh(self.is2D)
        self.model = PINN(input_shape=2, output_shape=3, eq = self.problemTag, layers=[20,40,60,40,20])

        self.loss = None
        self.Plot = None
        
        self.xRange = xRange
        self.yRange = yRange

        self.AoA = AoA
        self.c   = 1.0  #Airfoil Chord
        self.x0  = 0.0  #Airfoil leading edge x coordinate
        self.y0  = 0.0  #Airfoil leading edge y coordinate

        self.generate_airfoil_coords()

        return
    
    def generate_airfoil_coords(self, N=100, thickness=0.12):
        """
        Generate NACA 4-digit airfoil coordinates for both upper and lower surfaces.
        The lower surface is just the negative of the upper surface.
        """
        # Generate x coordinates
        x = np.linspace(self.x0, self.x0 + self.c, N)
        x_normalized = x / self.c
        
        # Generate upper surface
        y_upper = self.y0 + 5 * thickness * (0.2969 * np.sqrt(x_normalized) 
                                - 0.1260 * x_normalized 
                                - 0.3516 * x_normalized**2 
                                + 0.2843 * x_normalized**3 
                                - 0.1015 * x_normalized**4)
        
        # Generate lower surface
        y_lower = self.y0 - y_upper
        
        # Combine coordinates in the correct order (counterclockwise)
        self.xAirfoil = np.concatenate([x, np.flip(x)]).reshape(-1, 1)
        self.yAirfoil = np.concatenate([y_upper, np.flip(y_lower)]).reshape(-1, 1)
        
        return
    
    def generateMesh(self, Nx=100, Ny=100, NBoundary=100, sampling_method='random'):
        # Initialize boundaries
        self.mesh.boundaries = {
            'Inlet': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'Outlet': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'bottom': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'top': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None}
        }

        self.mesh.interiorBoundaries = {
            'Airfoil': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None}
        }
        
        self.mesh.setBoundary('top',
                    np.linspace(self.xRange[0], self.xRange[1], NBoundary),
                    np.full((NBoundary, 1), self.yRange[1], dtype=np.float32),
                    u = np.ones(NBoundary), v = np.zeros(NBoundary))

        self.mesh.setBoundary('bottom',
                    np.linspace(self.xRange[0], self.xRange[1], NBoundary),
                    np.full((NBoundary, 1), self.yRange[0], dtype=np.float32),
                    u = np.ones(NBoundary), v = np.zeros(NBoundary))

        self.mesh.setBoundary('Inlet',
                    np.full((NBoundary, 1), self.xRange[0], dtype=np.float32),
                    np.linspace(self.yRange[0], self.yRange[1], NBoundary),
                    u = np.ones(NBoundary), v = np.zeros(NBoundary))

        self.mesh.setBoundary('Outlet',
                    np.full((NBoundary, 1), self.xRange[1], dtype=np.float32),
                    np.linspace(self.yRange[0], self.yRange[1], NBoundary),
                    u = np.ones(NBoundary), v = np.zeros(NBoundary))
        
        self.mesh.setBoundary('Airfoil',
                    self.xAirfoil,
                    self.yAirfoil,
                    u = np.zeros(NBoundary), 
                    v = np.zeros(NBoundary),
                    interior=True)
        
        # Generate the mesh
        self.mesh.generateMesh(
            Nx=Nx,
            Ny=Ny,
            sampling_method=sampling_method
        )
        return
    
    def getLossFunction(self):
        self.loss = NavierStokesLoss(self.mesh, self.model)
    
    def train(self, epochs=10000, print_interval=100,  autosaveInterval=10000):
        self.getLossFunction()
        self.model.train(self.loss.loss_function, epochs=epochs, print_interval=print_interval,autosave_interval=autosaveInterval)

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
        self.Plot.scatterPlot(solkey)
