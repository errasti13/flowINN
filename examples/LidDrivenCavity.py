from src.tests.LidDrivenCavity import LidDrivenCavity

def main():
    x_range = (-1, 1)
    y_range = (-1, 1)

    equation = LidDrivenCavity(x_range, y_range)
    
    equation.generateMesh()

    return

if __name__ == "__main__":
    main()