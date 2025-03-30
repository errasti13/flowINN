from flowinn.tests.UnsteadyCylinder import UnsteadyCylinder

def main():
    # Define simulation parameters
    case_name = "UnsteadyCylinder"
    x_range = (-4.0, 8.0)  # Domain length
    y_range = (-4.0, 4.0)  # Domain height
    t_range = (0.0, 10.0)  # Time range - shorter time period for better resolution
    
    # Training parameters
    epochs = 30000        # More epochs for better convergence
    print_interval = 100  # Less frequent printing
    autosave_interval = 1000
    
    # Mesh parameters
    nx = 70
    ny = 70
    nt = 150              # Increased time discretization
    n_boundary = 300
    num_batches = 16      # More batches for better sampling

    trainedModel = False
    
    try:
        # Initialize airfoil flow
        cylinder = UnsteadyCylinder(case_name, x_range, y_range, t_range)
        
        # Generate mesh
        print("Generating mesh...")
        cylinder.generateMesh(Nx=nx, Ny=ny, Nt=nt,
                           NBoundary=n_boundary, 
                           sampling_method='random')

        # Train the model
        if trainedModel:
            print("Loading pre-trained model...")
            cylinder.load_model()
        else:
            print("Starting training...")
            cylinder.train(epochs=epochs, 
                         print_interval=print_interval,
                         autosaveInterval=autosave_interval,
                         num_batches=num_batches)
        
        # Predict and visualize
        print("Predicting flow field...")
        cylinder.predict()
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()