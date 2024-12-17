
from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load the model
filepath = '/workspaces/Plant-Leaf-Disease-Prediction/model.h5'
model = load_model(filepath)
print("Model Loaded Successfully")

# Prediction function
def pred_tomato_disease(tomato_plant):
    test_image = load_img(tomato_plant, target_size=(128, 128))  # Load image
    print("@@ Got Image for prediction")
  
    test_image = img_to_array(test_image) / 255.0  # Convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # Change dimension 3D to 4D
  
    result = model.predict(test_image)  # Predict diseased plant or not
    print('@@ Raw result = ', result)
  
    pred = np.argmax(result, axis=1)[0]  # Extract predicted class index
    print(pred)
    if pred == 0:
        return "Tomato - Bacteria Spot Disease", 'Tomato-Bacteria Spot.html'
    elif pred == 1:
        return "Tomato - Early Blight Disease", 'Tomato-Early_Blight.html'
    elif pred == 2:
        return "Tomato - Healthy and Fresh", 'Tomato-Healthy.html'
    elif pred == 3:
        return "Tomato - Late Blight Disease", 'Tomato-Late_blight.html'
    elif pred == 4:
        return "Tomato - Leaf Mold Disease", 'Tomato-Leaf_Mold.html'
    elif pred == 5:
        return "Tomato - Septoria Leaf Spot Disease", 'Tomato-Septoria_leaf_spot.html'
    elif pred == 6:
        return "Tomato - Target Spot Disease", 'Tomato-Target_Spot.html'
    elif pred == 7:
        return "Tomato - Tomato Yellow Leaf Curl Virus Disease", 'Tomato-Tomato_Yellow_Leaf_Curl_Virus.html'
    elif pred == 8:
        return "Tomato - Tomato Mosaic Virus Disease", 'Tomato-Tomato_mosaic_virus.html'
    elif pred == 9:
        return "Tomato - Two Spotted Spider Mite Disease", 'Tomato-Two-spotted_spider_mite.html'

# Create Flask instance
app = Flask(__name__)

# Render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Get input image from client, predict class, and render respective .html page for solution
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # Fetch input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        # Save the file to the static/uploads folder
        upload_folder = 'static/upload/'
        os.makedirs(upload_folder, exist_ok=True)  # Create folder if it doesn't exist
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_tomato_disease(tomato_plant=file_path)
              
        return render_template(output_page, pred_output=pred, user_image=file_path)

# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=True, port=8080)
