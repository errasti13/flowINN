from flowinn.tests.UnsteadyCylinder import UnsteadyCylinder
import argparse
import os
import tensorflow as tf

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unsteady Cylinder Flow Simulation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--print-interval', type=int, default=1, help='Print interval during training')
    parser.add_argument('--autosave-interval', type=int, default=1000, help='Model autosave interval')
    parser.add_argument('--num-batches', type=int, default=1, help='Number of batches per epoch')
    parser.add_argument('--load-model', action='store_true', help='Load pre-trained model')
    parser.add_argument('--predict-only', action='store_true', help='Skip training and only run prediction')
    parser.add_argument('--spatial-batches', type=int, default=1, help='Number of spatial batches')
    parser.add_argument('--temporal-batches', type=int, default=1, help='Number of temporal batches')
    parser.add_argument('--architecture', choices=['mlp', 'mfn', 'fast_mfn'], default='mfn',
                       help='Neural network architecture to use')
    
    args = parser.parse_args()
    
    # Define simulation parameters
    case_name = "UnsteadyCylinder"
    x_range = (-4.0, 8.0)  # Domain length
    y_range = (-4.0, 4.0)  # Domain height
    t_range = (0.0, 40.0)  # Time range - shorter time period for better resolution
    
    # Training parameters from command line args
    epochs = args.epochs
    print_interval = args.print_interval
    autosave_interval = args.autosave_interval
    num_batches = args.num_batches
    architecture = args.architecture
    
    # Memory management parameters
    spatial_batches = args.spatial_batches
    temporal_batches = args.temporal_batches

    nx = 70
    ny = 70
    nt = 150
    
    n_boundary = 100
    
    try:
        # Initialize cylinder flow
        print(f"\nInitializing UnsteadyCylinder with {architecture} architecture...")
            
        cylinder = UnsteadyCylinder(
            case_name, 
            x_range, y_range, t_range,
            activation='gelu',
            architecture=architecture
        )
        
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
                num_spatial_batches=spatial_batches,
                num_temporal_batches=temporal_batches
            )
            
            # Save model after training
            print("Training complete, saving model...")
            cylinder.save_model()
        
        # Start prediction
        print("\nStarting prediction...")
            
        cylinder.predict()
        cylinder.animate_flow()
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()