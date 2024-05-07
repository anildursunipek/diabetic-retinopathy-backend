import base64
from PIL import Image
import numpy as np
from io import BytesIO

image_path = "samples/moderate/0c55d58bebaf.png"
image = Image.open(image_path)

# Preprocess the image
image_array = np.array(image)

# Open the image to check result
image_pil = Image.fromarray(image_array)
# image_pil.show()
print("[INFO] Type of Image: ", type(image_array))
print("[INFO] Image Shape: ", image_array.shape)

# Convert the image to base64 format
buffered = BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
data_url = f'data:image/png;base64,{img_str}'

print(data_url[:100])
image_data = base64.b64decode(img_str)
image = Image.open(BytesIO(image_data))
image.show()