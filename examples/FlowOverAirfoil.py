import numpy as np
from flowinn.tests.FlowOverAirfoil import FlowOverAirfoil

def main():
    # Define simulation parameters
    case_name = "naca0012"
    x_range = (-2.0, 4.0)  # Domain length
    y_range = (-2.0, 2.0)  # Domain height
    angle_of_attack = 10.0  # Degrees
    
    # Training parameters
    epochs = 100
    print_interval = 100
    autosave_interval = 10000
    
    # Mesh parameters
    nx = 80
    ny = 60
    n_boundary = 200
    num_batches = 10

    trainedModel = False
    
    try:
        # Initialize airfoil flow
        airfoil = FlowOverAirfoil(case_name, x_range, y_range, angle_of_attack)
        
        # Generate mesh
        print("Generating mesh...")
        airfoil.generateMesh(Nx=nx, Ny=ny, 
                           NBoundary=n_boundary, 
                           sampling_method='uniform')
        airfoil.mesh.showMesh()

        # Train the model
        if trainedModel:
            print("Loading pre-trained model...")
            airfoil.load_model()
        else:
            print("Starting training...")
            airfoil.train(epochs=epochs, 
                         print_interval=print_interval,
                         autosaveInterval=autosave_interval,
                         num_batches=num_batches)
        
        # Predict and visualize
        print("Predicting flow field...")
        airfoil.predict()
        
        # Create various plots
        print("\nGenerating plots...")
        
        # 1. Default scatter plots
        print("Creating scatter plots...")
        airfoil.plot(solkey='u')  # Velocity in x-direction
        airfoil.plot(solkey='v')  # Velocity in y-direction
        airfoil.plot(solkey='p')  # Pressure field
        airfoil.plot(solkey='vMag')  # Velocity magnitude
        
        print("Simulation and visualization completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()