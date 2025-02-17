from flowinn.tests.LidDrivenCavity import LidDrivenCavity

def main():
    # Domain setup
    x_range = (-1, 1)
    y_range = (-1, 1)
    
    # Simulation parameters
    case_name = "LidDrivenCavity"
    epochs = 10000
    print_interval = 100
    autosave_interval = 10000
    
    
    # Mesh parameters
    nx = 200
    ny = 200
    n_boundary = 100
    num_batches = 5

    trainedModel = False
    
    try:
        # Initialize simulation
        cavity = LidDrivenCavity(case_name, x_range, y_range)
        
        # Generate mesh
        print("Generating mesh...")
        cavity.generateMesh(Nx=nx, Ny=ny, NBoundary=n_boundary, sampling_method='uniform')
        
        # Train or load model
        if trainedModel:
            print("Loading pre-trained model...")
            cavity.load_model()
        else:
            print("Starting training...")
            cavity.train(epochs=epochs, 
                        print_interval=print_interval,
                        autosaveInterval=autosave_interval,
                        num_batches=num_batches)  # Add num_batches parameter
        
        # Predict and visualize
        print("Predicting flow field...")
        cavity.predict()
        
        # Plot results
        print("Generating plots...")
        cavity.plot(solkey='u')
        cavity.plot(solkey='v')
        cavity.plot(solkey='p')
        
        print("Simulation completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()