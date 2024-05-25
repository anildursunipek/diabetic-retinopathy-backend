from flask import Flask, jsonify, request
from PIL import Image
from flask_cors import CORS
import time
import numpy as np
import base64
from io import BytesIO
import gc

# Our classes
from model import DetectionModel
from preprocessor import Preprocessor

app = Flask(__name__)
CORS(app, resources={r"/dr/image/upload": {"origins": "http://localhost:4200"}})

global binaryModel
global multiClassModel
global preprocessor

binaryModel = DetectionModel(model_path="saved_models/efficientnetb4_binary_exported_model", threshold=0.5)
multiClassModel = DetectionModel(model_path="saved_models/efficientnetb4_multiclass_exported_model")
preprocessor = Preprocessor(image_size=380)

@app.route("/status", methods=['GET'])
def checkStatus():
    return jsonify({'Status': 'System online'}), 200

@app.route("/dr/image/upload", methods=['POST'])
def upload_image():
    start_time = time.perf_counter()
    data = request.get_json()

    if 'base64image' not in data:
        return jsonify({'error': 'Base64image not found in request body'}), 400

    if "filename" not in data or data["filename"] == "":
        return jsonify({'error': 'No selected file or filename is empty'}), 400

    base64_image = data.get('base64image')
    filename = data.get('filename')

    # Convert base64 image to array
    try:
        image_data = base64.b64decode(base64_image.split(',')[1])
        image = Image.open(BytesIO(image_data))
        # image.show()

        # Preprocess the image
        image_array = np.array(image)
        preprocessed_image = preprocessor.preprocessing(image_array)

        # Convert preprocessed_image to base64
        image_pil = Image.fromarray(preprocessed_image)
        # image_pil.show()

        buffered = BytesIO()
        image_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        preprocessed_base64_image = f'data:image/png;base64,{img_str}'

        # Make a prediction
        inference_start = time.perf_counter()

        # Binary Classification  
        binary_prediction_result = binaryModel.prediction(image_array=preprocessed_image).numpy()
        binary_prediction_result = binary_prediction_result.tolist()[0]
        # print(binary_prediction_result[0])

        # Multi-class Classification
        if(binary_prediction_result[0] >= 0.5):
            multiclass_prediction_result = multiClassModel.prediction(image_array=preprocessed_image).numpy()
            multiclass_prediction_result = multiclass_prediction_result.tolist()[0]
            # print(multiclass_prediction_result[0])
        else:
            multiclass_prediction_result = [[0.0, 0.0, 0.0, 0.0]]

        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start

        end_time = time.perf_counter()
        response_time = end_time - start_time
        gc.collect()

        return jsonify({
            'original_image' : base64_image,
            'message': 'Image received', 
            'response_time' : response_time, # float
            'filename': filename, # string
            'inference_time': inference_time, # float
            'binary_prediction_result' : binary_prediction_result, # list
            'preprocessed_image' : preprocessed_base64_image, # base64
            'multiclass_prediction_result' : multiclass_prediction_result # list
            }), 200
    
    except Exception as e:
        print(e)
        gc.collect()
        return jsonify({'error': f'No image received {e}',}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)