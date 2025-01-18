from src.tests.LidDrivenCavity import LidDrivenCavity


def main():
    x_range = (-1, 1)
    y_range = (-1, 1)

    equation = LidDrivenCavity('LidDrivenCavity', x_range, y_range)
    
    equation.generateMesh(Nx = 60, Ny = 100)

    #equation.train(epochs=1000)

    equation.model.load("trainedModels/" + equation.problemTag + ".tf")

    equation.predict()
    equation.plot('u')
    equation.plot('v')
    equation.plot('p')

    return

if __name__ == "__main__":
    main()