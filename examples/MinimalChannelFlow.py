import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from src.tests.MinimalChannelFlow import MinimalChannelFlow

def main():
    # Define simulation parameters
    case_name = "channel_flow_3d"
    x_range = (0.0, 3.0)  # Channel length
    y_range = (0.0, 1.0)   # Channel height
    z_range = (0.0, 1.0)   # Channel width
    
    # Training parameters
    epochs = 20000
    print_interval = 100
    autosave_interval = 5000

    # Mesh parameters
    nx = 40
    ny = 30
    nz = 30
    n_boundary = 100
    
    try:
        # Initialize channel flow
        channel = MinimalChannelFlow(case_name, x_range, y_range, z_range)
        
        # Generate mesh
        print("Generating mesh...")
        channel.generateMesh(Nx=nx, Ny=ny, Nz=nz, 
                           NBoundary=n_boundary, 
                           sampling_method='uniform')
        
        channel.mesh.showMesh()
        
        # Train the model
        print("Starting training...")
        channel.train(epochs=epochs, 
                     print_interval=print_interval,
                     autosaveInterval=autosave_interval)
        
        # Predict and visualize
        print("Predicting flow field...")
        channel.predict()
        
        # Plot results
        print("Generating plots...")
        channel.plot(solkey='u')  # Velocity in x-direction
        channel.plot(solkey='v')  # Velocity in y-direction
        channel.plot(solkey='w')  # Velocity in z-direction
        channel.plot(solkey='p')  # Pressure field
        
        print("Simulation completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
