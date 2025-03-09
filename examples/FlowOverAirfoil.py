from flowinn.tests.FlowOverAirfoil import FlowOverAirfoil

def main():
    # Define simulation parameters
    case_name = "naca0012"
    x_range = (-4.0, 4.0)  # Domain length
    y_range = (-2.0, 2.0)  # Domain height
    angle_of_attack = 10.0  # Degrees
    
    # Training parameters
    epochs = 10000
    print_interval = 100
    autosave_interval = 10000
    
    # Mesh parameters
    nx = 200
    ny = 200
    n_boundary = 200
    num_batches = 5

    trainedModel = False
    
    try:
        # Initialize airfoil flow
        airfoil = FlowOverAirfoil(case_name, x_range, y_range, angle_of_attack)
        
        # Generate mesh
        print("Generating mesh...")
        airfoil.generateMesh(Nx=nx, Ny=ny, 
                           NBoundary=n_boundary, 
                           sampling_method='random')

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
        
        # Vector field visualization
        print("Creating vector field plot...")
        airfoil.plot(plot_type='quiver', scale=30, density=1)
        
        print("Simulation and visualization completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()