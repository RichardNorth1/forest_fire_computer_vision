from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
from pathlib import Path

__version__ = '0.1.0'

BASE_DIR = Path(__file__).resolve(strict=True).parent
# Load the trained model
model = load_model(BASE_DIR / f'cnn_model-{__version__}.h5')

async def predict_pipeline(file):
    image = Image.open(io.BytesIO(await file.read()))
    # Preprocess the image
    image = image.resize((128, 128))  # Resize to the same size used during training
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image)
    is_fire = prediction[0][0] > 0.5  # Assuming binary classification with sigmoid activation
    return is_fire