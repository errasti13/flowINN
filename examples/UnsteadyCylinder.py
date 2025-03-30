from flowinn.tests.UnsteadyCylinder import UnsteadyCylinder
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unsteady Cylinder Flow Simulation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--print-interval', type=int, default=100, help='Print interval during training')
    parser.add_argument('--autosave-interval', type=int, default=1000, help='Model autosave interval')
    parser.add_argument('--num-batches', type=int, default=2, help='Number of batches per epoch')
    parser.add_argument('--use-cpu', action='store_true', help='Force CPU usage for training only')
    parser.add_argument('--load-model', action='store_true', help='Load pre-trained model')
    parser.add_argument('--predict-only', action='store_true', help='Skip training and only run prediction')
    
    args = parser.parse_args()
    
    # Define simulation parameters
    case_name = "UnsteadyCylinder"
    x_range = (-4.0, 8.0)  # Domain length
    y_range = (-4.0, 4.0)  # Domain height
    t_range = (0.0, 10.0)  # Time range - shorter time period for better resolution
    
    # Training parameters from command line args
    epochs = args.epochs
    print_interval = args.print_interval
    autosave_interval = args.autosave_interval
    num_batches = args.num_batches
    use_cpu = args.use_cpu
    
    # Mesh parameters
    nx = 70
    ny = 70
    nt = 150              # Increased time discretization for better capture of vortex shedding
    n_boundary = 300
    
    if use_cpu:
        print("\n*** USING CPU FOR TRAINING ONLY (NOT RECOMMENDED) ***")
    
    try:
        # Initialize cylinder flow
        cylinder = UnsteadyCylinder(case_name, x_range, y_range, t_range)
        
        # Generate mesh
        print("Generating mesh...")
        cylinder.generateMesh(Nx=nx, Ny=ny, Nt=nt,
                           NBoundary=n_boundary, 
                           sampling_method='random')

        # Train or load the model
        if args.load_model:
            print("Loading pre-trained model...")
            cylinder.load_model()
        elif not args.predict_only:
            print("Starting training...")
            cylinder.train(
                epochs=epochs, 
                print_interval=print_interval,
                autosaveInterval=autosave_interval,
                num_batches=num_batches,
                use_cpu=use_cpu
            )
        
        # Predict and visualize
        if not args.load_model and not args.predict_only:
            print("Training complete, saving model...")
            # Additional save just to be safe
            import os
            os.makedirs('trainedModels', exist_ok=True)
            cylinder.model.model.save(f'trainedModels/{case_name}.keras')
        
        print("Predicting flow field on same device as training...")
        cylinder.predict(use_cpu=False)  # Never force CPU for prediction - use GPU
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()