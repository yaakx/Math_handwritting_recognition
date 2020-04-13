import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from modules.my_class import ImageSolver
from modules.resolve import MathSolver

app = Flask(__name__)


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            filename = 'static/' + secure_filename(image.filename)
            image.save(filename)
            imsol = ImageSolver(filename, os.getcwd() + '/data/models/third_model.h5')
            msol = MathSolver(imsol.equation, imsol.numbers.white)
            return render_template('image.html', text=msol)
    return render_template('test.html')


if __name__ == "__main__":
    app.run(threaded=False)
