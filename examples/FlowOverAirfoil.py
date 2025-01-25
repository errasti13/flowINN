from src.tests.LidDrivenCavity import LidDrivenCavity


def main():
    x_range = (0, 1)
    y_range = (0, 1)

    equation = LidDrivenCavity('LidDrivenCavity', x_range, y_range)
    
    equation.generateMesh(Nx = 60, Ny = 100, sampling_method='random')

    equation.train(epochs=100000)

    equation.model.load(equation.problemTag)

    equation.predict()
    equation.plot(solkey='vMag', streamlines=True)
    equation.plot(solkey='u', streamlines=True)
    equation.plot(solkey='v', streamlines=True)
    equation.plot(solkey='p', streamlines=True)

    return

if __name__ == "__main__":
    main()