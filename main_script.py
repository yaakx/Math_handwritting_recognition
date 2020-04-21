import argparse
import os
from app.modules.my_class import ImageSolver
from app.modules.resolve import MathSolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Upload an image to resolve.",
                        type=str)
    args = parser.parse_args()
    try:
        imsol = ImageSolver(args.image, os.getcwd() + '/app/models/models/third_model.h5')
        msol = MathSolver(imsol.equation)
        print(imsol)
        if str(imsol) != str(msol.solution):
            print(msol.solution)

    except:
        print("Could't resolve equation or invalid archive.")


if __name__ == '__main__':
    main()