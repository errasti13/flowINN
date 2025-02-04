from src.tests.FlowOverAirfoil import FlowOverAirfoil

def main():
    # Domain setup
    x_range = (-3.0, 5.0)
    y_range = (-3.0, 3.0)
    angle_of_attack = 5.0  # degrees
    
    # Simulation parameters
    case_name = "NACA0012"
    epochs = 20000
    print_interval = 100
    autosave_interval = 5000
    
    # Mesh parameters
    nx = 40
    ny = 40
    n_boundary = 100
    
    try:
        # Initialize simulation
        airfoil = FlowOverAirfoil(case_name, x_range, y_range, AoA=angle_of_attack)
        
        # Generate mesh
        print("Generating mesh...")
        airfoil.generateMesh(Nx=nx, Ny=ny, NBoundary=n_boundary, sampling_method='random')
        
        # Train the model
        print("Starting training...")
        airfoil.train(epochs=epochs, 
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