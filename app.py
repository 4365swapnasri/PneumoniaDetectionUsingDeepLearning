from flask import Flask, request, render_template
import numpy as np
import cv2
import os
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl","rb")) # Ensure the model is trained for binary classification


UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (150, 150))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values (0-1)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (150,150,1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1,150,150,1)
    
    return img
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")
@app.route("/about", methods=["GET", "POST"])
def about():
    return render_template("about.html")
@app.route("/contact", methods=["GET", "POST"])
def contact():
    return render_template("contact.html")
#@app.route("/", methods=["GET", "POST"])
#def project():
 #   return render_template("project.html")
@app.route("/project", methods=["GET", "POST"])
def upload_file():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        if "image" not in request.files:
            prediction = "No file uploaded"
        else:
            file = request.files["image"]
            if file.filename == "":
                prediction = "No selected file"
            else:
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(file_path)

                img = preprocess_image(file_path)
                pred_prob = model.predict(img)[0][0]

                if pred_prob >= 0.5:
                    prediction = "Normal"
                    confidence = pred_prob * 100
                else:
                    prediction = "Pneumonia"
                    confidence = (1 - pred_prob) * 100

                image_path = file_path

    return render_template("project.html", prediction=prediction, confidence=confidence, image_path=image_path)
"""@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    if image:
        # You can save or process the image here
        filename = image.filename
        # image.save(os.path.join('static/uploads', filename))  # Optional

        # Dummy result (replace with model prediction)
        result = "Pneumonia Detected"

        return render_template("result.html", prediction=result, filename=filename)"""
@app.route("/predict", methods=["POST"])
def predict():
    #if request.method == "POST":
    #    if "image" not in request.files:
    #       return "No file uploaded"
        
    file = request.files["image"]
    if file.filename == "":
        return "No selected file"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    img = preprocess_image(file_path)
    prediction = model.predict(img)[0][0]  # Get the prediction probability
        # Convert probability to label
    if prediction >= 0.5:
        predicted_label = "Normal"
        confidence = prediction * 100  # Confidence in %
        return render_template("index1.html", prediction=predicted_label, confidence=confidence, image_path=file_path)
    else:
        predicted_label = "Pneumonia"
        confidence = (1 - prediction) * 100  # Confidence in %
        return render_template("index1.html", prediction=predicted_label, confidence=confidence, image_path=file_path)

    #return render_template("index1.html", prediction=None, confidence=None, image_path=None)'''


#@app.route("/predict", methods=["GET", "POST"])
#def predict():
#    return render_template("predict.html")


if __name__ == "__main__":
    app.run(port=3000,debug=True)