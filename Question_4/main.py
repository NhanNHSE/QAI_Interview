from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import numpy as np
import pickle
from PIL import Image
import io

app = FastAPI()

# Load the trained model parameters
with open('mnist_model.pkl', 'rb') as f:
    W, b = pickle.load(f)

def preprocess_image(image_file):
    try:
        img = Image.open(image_file).convert('L')
        img = img.resize((28, 28))  # Assuming the model expects 28x28 images
        img_array = np.array(img).reshape(-1)
        img_array = img_array / 255.0
        return img_array.reshape(-1, 1)  # Reshape for compatibility with model input
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def predict(W, b, X):
    Z = np.dot(W, X) + b
    A = sigmoid(Z)
    return A

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FastAPI App</title>
    </head>
    <body>
        <h1>Welcome to the FastAPI application!</h1>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <input type="submit" value="Upload Image">
        </form>
    </body>
    </html>
    """

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        img_array = preprocess_image(file.file)
        A = predict(W, b, img_array)
        prediction = np.argmax(A)
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
