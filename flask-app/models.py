import numpy as np
import tensorflow as tf
import time
import sys
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# Custom libraries
from preprocessor import Preprocessor

class BinaryModel:
    def __init__(self, model_path: str, threshold:float = 0.50):
        self.model_path = model_path
        self.threshold = threshold

        # Load the saved model
        self.model = tf.keras.models.load_model(model_path)



    def prediction(self, image_array: np.array):
        image_array = np.expand_dims(image_array, axis=0)
        result = self.model.predict(image_array)
        return result

class MultiClassModel:
    def __init__(self, model_path: str):
        self.model_path = model_path

        # Load the saved model
        self.model = tf.keras.models.load_model(model_path)

    def prediction(self, image_array: np.array):
        image_array = np.expand_dims(image_array, axis=0)
        result = self.model.predict(image_array)
        return result

if __name__ == "__main__":
    # Test Codes
    start = time.perf_counter() # Start timer to see total run time

    # Print current tensorflow version
    print("[INFO] Version of TF: ", tf.version.VERSION)

    # Make a prediction from sample folder

    # Read image from folder
    image_path = "samples/moderate/0c55d58bebaf.png"
    image = Image.open(image_path)

    # Preprocess the image
    image_array = np.array(image)
    preprocessor = Preprocessor(image_size=528)
    preprocessed_image = preprocessor.preprocessing(image_array)

    # Open the image to check result
    image_pil = Image.fromarray(preprocessed_image)
    image_pil.show()
    print("[INFO] Type of Image: ", type(image_array))
    print("[INFO] Image Shape: ", image_array.shape)

    # Create models
    binaryModel = BinaryModel(model_path="saved_models/dr_binary_model", threshold=0.5)
    multiClassModel = MultiClassModel(model_path="saved_models/dr_multiclass_model")

    # Make a prediction
    inference_start = time.perf_counter()

    prediction_result = binaryModel.prediction(image_array=preprocessed_image)
    prediction_result = prediction_result.tolist()

    multiclass_prediction_result = multiClassModel.prediction(image_array=preprocessed_image)
    multiclass_prediction_result = multiclass_prediction_result.tolist()

    print("[INFO] Result of binary model prediction: ", prediction_result)
    print("[INFO] Type of prediction result: ", type(prediction_result))

    print("[INFO] Result of multiclass model prediction: ", multiclass_prediction_result)
    print("[INFO] Type of prediction result: ", type(multiclass_prediction_result))

    inference_end = time.perf_counter()
    print("[INFO] Total inference time: ", inference_end - inference_start, " seconds")

    ## END OF THE PROGRAM
    end = time.perf_counter()

    # Check the active gpu devices
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Print informations
    print("[INFO] Total run time: ", end - start, " seconds")
