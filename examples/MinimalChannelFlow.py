import numpy as np
from flowinn.tests.MinimalChannelFlow import MinimalChannelFlow

def main():
    # Define simulation parameters
    case_name = "channel_flow_3d"
    x_range = (0.0, 3.0)  # Channel length
    y_range = (0.0, 1.0)   # Channel height
    z_range = (0.0, 1.0)   # Channel width
    
    # Training parameters
    epochs = 100
    print_interval = 100
    autosave_interval = 10000
    
    # Mesh parameters
    nx = 50
    ny = 30
    nz = 50
    num_batches = 10
    n_boundary = 100

    trainedModel = False
    
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
        if trainedModel:
            print("Loading pre-trained model...")
            channel.load_model()
        else:
            print("Starting training...")
            channel.train(epochs=epochs, 
                        print_interval=print_interval,
                        autosaveInterval=autosave_interval,
                        num_batches=num_batches)
        
        # Predict and visualize
        print("Predicting flow field...")
        channel.predict()
        
        # Create various plots
        print("\nGenerating plots...")
        
        # 1. Default scatter plots
        print("Creating scatter plots...")
        channel.plot(solkey='u')  # Velocity in x-direction
        channel.plot(solkey='v')  # Velocity in y-direction
        channel.plot(solkey='w')  # Velocity in z-direction
        channel.plot(solkey='p')  # Pressure field
        channel.plot(solkey='vMag')  # Velocity magnitude
        
        # 2. 3D scatter plots
        print("Creating 3D scatter plots...")
        channel.plot(solkey='vMag', plot_type='3d')
        
        # 3. Z-plane slices
        print("Creating slice plots...")
        channel.plot(solkey='vMag', plot_type='slices', 
                    z_cuts=[0.25, 0.5, 0.75],
                    num_points=100)
        
        print("Simulation and visualization completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
