from flowinn.tests.FlowOverAirfoil import FlowOverAirfoil

def main():
    # Domain setup
    x_range = (-3.0, 5.0)
    y_range = (-3.0, 3.0)
    angle_of_attack = 5.0  # degrees
    
    # Simulation parameters
    case_name = "FlowOverAirfoil"
    epochs = 1000
    print_interval = 100
    autosave_interval = 1000
    
    # Mesh parameters
    nx = 200
    ny = 200
    n_boundary = 100
    num_batches = 5

    trainedModel = False
    
    try:
        # Initialize simulation
        airfoil = FlowOverAirfoil(case_name, x_range, y_range, AoA=angle_of_attack)

        # Generate mesh
        print("Generating mesh...")
        airfoil.generateMesh(Nx=nx, Ny=ny, NBoundary=n_boundary, sampling_method='random')
        airfoil.mesh.showMesh()
        
        # Train the model
        if trainedModel:
            print("Loading pre-trained model...")
            airfoil.load_model()
        else:
            print("Starting training...")
            airfoil.train(epochs=epochs, 
                         num_batches=num_batches,  # Changed from batch_size
                         print_interval=print_interval,
                         autosaveInterval=autosave_interval)
        
        # Predict and visualize
        print("Predicting flow field...")
        airfoil.predict()
        
        # Plot results
        print("Generating plots...")
        airfoil.plot(solkey='u')
        airfoil.plot(solkey='v')
        airfoil.plot(solkey='p')
        
        print("Simulation completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()