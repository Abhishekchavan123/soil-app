import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your model
model = load_model(r"C:\Users\Admin\Downloads\mobilenetv2_soil_finetuned_final.h5", compile=False)

# ✅ FIXED: Keep class labels consistent with training
# If during training: train_gen.class_indices = {'High':0,'Low':1,'Medium':2}
labels = ["High", "Low", "Medium"]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0  # same as training
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_url = None
    
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            # Preprocess and predict
            img_tensor = preprocess_image(filepath)
            preds = model.predict(img_tensor)[0]

            predicted_index = np.argmax(preds)
            prediction = labels[predicted_index]
            confidence = preds[predicted_index] * 100
            image_url = filepath

    # return render_template('index.html', prediction=prediction, confidence=confidence, image_url=image_url)
            all_preds = {labels[i]: float(preds[i]*100) for i in range(len(labels))}
    return render_template(
    'index.html',
    prediction=prediction,
    confidence=confidence,
    image_url=image_url,
    all_preds=all_preds if 'all_preds' in locals() else {}
)


if __name__ == '__main__':
    app.run(debug=True)
