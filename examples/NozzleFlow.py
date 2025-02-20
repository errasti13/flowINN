from flowinn.tests.FlowThroughNozzle import FlowThroughNozzle

def main():
    # Domain setup
    x_range = (0.0, 5.0)
    y_range = (-1.0, 1.0)
    
    # Simulation parameters
    case_name = "NozzleFlow"
    epochs = 1000
    print_interval = 100
    autosave_interval = 1000
    
    # Mesh parameters
    nx = 200
    ny = 200
    n_boundary = 150
    num_batches = 4

    trainedModel = False
    
    try:
        # Initialize simulation
        nozzle_flow = FlowThroughNozzle(case_name, x_range, y_range)

        # Generate mesh
        print("Generating mesh...")
        nozzle_flow.generateMesh(Nx=nx, Ny=ny, NBoundary=n_boundary, sampling_method='random')
        nozzle_flow.mesh.showMesh()
        
        # Train the model
        if trainedModel:
            print("Loading pre-trained model...")
            nozzle_flow.load_model()
        else:
            print("Starting training...")
            nozzle_flow.train(epochs=epochs, 
                         num_batches=num_batches,
                         print_interval=print_interval,
                         autosaveInterval=autosave_interval)
        
        # Predict and visualize
        print("Predicting flow field...")
        nozzle_flow.predict()
        
        # Plot results
        print("Generating plots...")
        nozzle_flow.plot(solkey='u')
        nozzle_flow.plot(solkey='v')
        nozzle_flow.plot(solkey='p')
        
        print("Simulation completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
