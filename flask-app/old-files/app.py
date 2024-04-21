from flask import Flask, jsonify, request
from PIL import Image
from flask_cors import CORS
import time
import numpy as np
import base64
from io import BytesIO

# Our classes
from prediction import BinaryModel
from preprocessor import Preprocessor

app = Flask(__name__)
CORS(app, resources={r"/dr/image/upload": {"origins": "http://localhost:4200"}})

global binaryModel
global preprocessor

binaryModel = BinaryModel(model_path="saved_models\dr_binary_model", threshold=0.5)
preprocessor = Preprocessor(image_size=528)


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
        image.show()

        end_time = time.time()
        response_time = end_time - start_time

        # Here you can perform the operations you want to do with the image.


        return jsonify({
            'message': 'Image received', 
            'response_time' : response_time,
            'filename': filename,
            }), 200
    
    except Exception as e:
        return jsonify({'error': 'No image received'}), 400

# @app.route("/dr/image/upload", methods=['POST'])
# def upload_image():
#     start_time = time.perf_counter()

#     if 'image' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['image']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Read Image with PIL
#     try:
#         img = Image.open(file)
#     except Exception as e:
#         return jsonify({'error': 'Could not open file', 'details': str(e)}), 500

#     # Here you can perform the operations you want to do with the image.
#     imageMatrix = np.array(img)
#     print(imageMatrix.shape)
#     img.show()
#     width, height = img.size
#     print("Widt: ", width, " Height: ", height)
#     end_time = time.perf_counter()
#     response_time = end_time - start_time

#     return jsonify({'filename': file.filename, 'width': width, 'height': height, 'response_time' : response_time}), 200


# @app.route("/dr/image/upload", methods=['POST'])
# def uploadImage():
#     start_time = time.time()

#     data = request.get_json()
#     base64_image = data.get('image')
    
#     if base64_image:
#         image_data = base64.b64decode(base64_image.split(',')[1])
#         image = Image.open(BytesIO(image_data))
#         image.show()
#         end_time = time.time()
#         response_time = end_time - start_time

#         # Here you can perform the operations you want to do with the image.
#         return jsonify({'message': 'Image received', 'response_time' : response_time}), 200
#     else:
#         return jsonify({'error': 'No image received'}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50100, debug=True)