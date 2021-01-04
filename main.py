from flask import Flask, render_template, Response, request, redirect, flash, url_for
from start_camera2 import VideoCamera,predict_image
import os

import urllib.request
#from app import app
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'static\\uploads'
SECRET_KEY = "32b9193967ebe896201ad136e21c430e027fb5f5e6fff7b0"


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = SECRET_KEY

@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/start_camera')
def start_camera():
    return render_template('video.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/submit_file', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],"temp.jpg"))
            filepath =os.path.join(app.config['UPLOAD_FOLDER'],"temp.jpg")
            #getPrediction(filename)
            #label, acc = getPrediction(filename)
            pred, label= predict_image.process_and_predict(filepath)

            if pred > 0.5:
                flash(pred)
                flash(filename)
                flash(label)
            else:
                pred = 100-pred
                flash(pred)
                flash(filename)
                flash(label)
            return redirect('/result')

@app.route('/display_image/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename= 'uploads/' + filename), code=301)

@app.route('/display_logo/<filename>')
def display_logo(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename= 'image/' + filename), code=301)


@app.route('/result')
def result():
    return render_template('result2.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if request.endpoint != "start_camera":
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
