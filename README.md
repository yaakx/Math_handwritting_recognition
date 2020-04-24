# Math handwritting recognition

## Overview
The projects consists on an app that recognizes mathematical symbols on an image and resolves the operation, ecuation or function in the image. This is done with a neural network that recognices each of the symbols in the image.

## Data
The data used is:
- [Handwirtten math symbol dataset](https://www.kaggle.com/xainano/handwrittenmathsymbol).
- Handwritten symbols from the author.

## Folder
```
└── project
    ├── .gitignore
    ├── requeriments.txt
    ├── README.md
    ├── main_script.py
    ├── notebooks
    │   ├── First_try.ipynb
    │   ├── Image_treatment.ipynb
    │   ├── New_model.ipynb
    │   ├── detect_ecuation.ipynb
    │   ├── resolve_sums.ipynb
    │   └── validate.ipynb
    └── app
        ├── app.py
        ├── models
        │   └── third_model.h5
        ├── static
        ├── templates
        └── modules
            ├── image_solver.py
            └── math_solver.py
```

## Tecnology
All the program is done in python and it uses mainly opencv for the image treatment, keras for creating, training and using the model and sympy for resolving the final solution. The app is done with flask.

## Usage
To run the app go to the 'app' folder and run `python app.py` then go to `localhost:5000` in your web browser and there you can upload images and it will show you what the neural network sees and the solution or graph in case of a function.

You can also run the program from terminal `python main_script.py -i ./path/to/image`.

If the app takes a long time to resolve it will probably have detected a very large power. It will stop alone and show you only what it has detected.

## Procedure
The program first binarices the image and separes it in each of its elements individually with opencv. Then rescalates each element and classifies it with the nerual network. Then with the result of all the elements firstly recognices powers and divisions looking at the original image and later tries to find the most probable result that is mathematically coherent.
Lastly using sympy it resolves the final result, if its a operation or ecuation its resolved and in case of a function it creates a graph using google charts.

The neural netwok has been created from scrath using keras for this project. The entrnace must be 45x45px and it has 18 possible exits that are all the symbols than it recognizes right now, this are: '0', 1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '(', ')', 'x', 'y', 'z' and  ','.

## Next steps
Make a better neural network by training it with different styles of handriwtting, also include more mathematical symbols like lim, log or 'e'.
Also treat better the image before dividing it so it can accept images woth more shadows and with several ecuations.
And lastly resolve better the ecuation so it can also resolve derivates and integrates and bigger and more complex ones.

