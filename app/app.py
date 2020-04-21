import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from modules.my_class import ImageSolver
from modules.resolve import MathSolver


app = Flask(__name__)
ALLOWED_EXTENSIONS = ['png', 'jpg']


def allowed_file(filename):
    return '.' in filename and \
           filename.split('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if allowed_file(image.filename):
                filename = 'static/uploads/' + 'eq' + secure_filename(image.filename)
                image.save(filename)
                try:
                    imsol = ImageSolver(filename, os.getcwd() + '/models/third_model.h5')
                    msol = MathSolver(imsol.equation)
                except:
                    return render_template('test.html')
                if msol.type == 'F':
                    if msol.solution.solution:
                        return render_template('function.html', array=msol.solution.solution, equation=msol.equation,
                                               filename=filename, symbol0=msol.solution.symbol[0], symbol1=msol.solution.symbol[1])
                    else:
                        return render_template('no_graph.html', equation=msol.equation, filename=filename)
                else:
                    return render_template('image.html', text=msol, equation=msol.equation, filename=filename)
            else:
                render_template('test.html')
    return render_template('test.html')


@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("upload_image"))


@app.route("/example", methods=["GET"])
def base():
    return render_template('examples.html')


if __name__ == "__main__":
    app.run(threaded=False, port=5000)
