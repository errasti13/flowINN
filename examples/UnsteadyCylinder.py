from flowinn.tests.UnsteadyCylinder import UnsteadyCylinder

def main():
    # Define simulation parameters
    case_name = "UnsteadyCylinder"
    x_range = (-4.0, 8.0)  # Domain length
    y_range = (-4.0, 4.0)  # Domain height
    t_range = (0.0, 10.0)  # Time range
    
    # Training parameters
    epochs = 10000
    print_interval = 100
    autosave_interval = 10000
    
    # Mesh parameters
    nx = 100
    ny = 100
    nt = 1000
    n_boundary = 300
    num_batches = 16

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