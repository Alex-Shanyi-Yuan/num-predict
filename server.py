from ctypes import resize
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO

import tensorflow as tf #tensor flow
import numpy as np
import cv2
import uvicorn


app = FastAPI()

@app.get('/')
def hello():
    return "hello"


@app.post('/api/predict')
def predict_img(file: UploadFile = File(...)):
    file_data = file.file.read()
    img = Image.open(BytesIO(file_data))

    #checking custom answer
    new_module = tf.keras.models.load_model("neural.h5")

    img = np.array(img)
    print(img)

    #loading and resizing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)
    newimg = tf.keras.utils.normalize(resized, axis=1) #normalize 0 to 1
    newimg = np.array(newimg).reshape(-1, 28, 28, 1)

    prediction = new_module.predict(newimg)
    return str(int(np.argmax(prediction)))

if __name__=="__main__":
    uvicorn.run(app, port=3000, host='localhost')

