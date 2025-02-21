import pytest
import numpy as np
from flowinn.tests.MinimalChannelFlow import MinimalChannelFlow
from flowinn.tests.LidDrivenCavity import LidDrivenCavity
from flowinn.tests.FlowThroughNozzle import FlowThroughNozzle
from flowinn.tests.FlowOverAirfoil import FlowOverAirfoil

def test_minimal_channel():
    try:
        case = MinimalChannelFlow(
            "test_channel",
            xRange=(0.0, 3.0),
            yRange=(0.0, 1.0),
            zRange=(0.0, 1.0)
        )
        case.generateMesh(Nx=20, Ny=20, Nz=20, NBoundary=50, sampling_method='uniform')
        case.mesh.showMesh()
        case.generateMesh(Nx=20, Ny=20, Nz=20, NBoundary=50, sampling_method='random')
        case.mesh.showMesh()
        case.getLossFunction()
        # Run a small number of epochs for testing
        case.train(epochs=10, print_interval=5, autosaveInterval=10, num_batches=2)
        case.predict()
        assert True
    except Exception as e:
        pytest.fail(f"MinimalChannelFlow test failed: {str(e)}")

def test_lid_driven_cavity():
    try:
        case = LidDrivenCavity(
            "test_cavity",
            xRange=(0.0, 1.0),
            yRange=(0.0, 1.0)
        )
        case.generateMesh(Nx=100, Ny=100, NBoundary=50, sampling_method='uniform')
        case.mesh.showMesh()
        case.generateMesh(Nx=100, Ny=100, NBoundary=50, sampling_method='random') 
        case.mesh.showMesh()
        case.getLossFunction()
        case.train(epochs=10, print_interval=5, autosaveInterval=10, num_batches=2)
        case.predict()
        assert True
    except Exception as e:
        pytest.fail(f"LidDrivenCavity test failed: {str(e)}")

def test_flow_through_nozzle():
    try:
        case = FlowThroughNozzle(
            "test_nozzle",
            xRange=(0.0, 3.0),
            yRange=(-1.0, 1.0)
        )
        case.generateMesh(Nx=100, Ny=100, NBoundary=50, sampling_method='uniform')
        case.mesh.showMesh()
        case.generateMesh(Nx=100, Ny=100, NBoundary=50, sampling_method='random')
        case.mesh.showMesh()
        case.getLossFunction()
        case.train(epochs=10, print_interval=5, autosaveInterval=10, num_batches=2)
        case.predict()
        assert True
    except Exception as e:
        pytest.fail(f"FlowThroughNozzle test failed: {str(e)}")

def test_flow_over_airfoil():
    try:
        case = FlowOverAirfoil(
            "test_airfoil",
            xRange=(-2.0, 4.0),
            yRange=(-2.0, 2.0),
            AoA=5.0
        )
        case.generateMesh(Nx=100, Ny=100, NBoundary=50, sampling_method='uniform')
        case.mesh.showMesh()
        case.generateMesh(Nx=100, Ny=100, NBoundary=50, sampling_method='random')
        case.mesh.showMesh()
        case.getLossFunction()
        case.train(epochs=10, print_interval=5, autosaveInterval=10, num_batches=2)
        case.predict()
        assert True
    except Exception as e:
        pytest.fail(f"FlowOverAirfoil test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])
