from src.tests.LidDrivenCavity import LidDrivenCavity

def main():
    x_range = (-1, 1)
    y_range = (-1, 1)

    equation = LidDrivenCavity('LidDrivenCavity', x_range, y_range)
    
    equation.generateMesh(Nx = 200, Ny = 200)

    equation.train(epochs=1000)

    equation.predict()
    equation.plot()

    return

if __name__ == "__main__":
    main()