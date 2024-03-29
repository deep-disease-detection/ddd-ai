from fastapi import FastAPI
from ddd.ml_logic.registry import load_model
import tensorflow as tf
import numpy as np
from ddd.params import *
import base64
from imageio import imread
import io
from pydantic import BaseModel

app = FastAPI()
app.state.model = load_model()
print("model ready")


@app.get("/")
def root():
    return {"Hello": "World"}


class Item(BaseModel):
    image: str


@app.post("/predict/")
def predict(post_dict: Item):
    b64code = post_dict.image
    # print(b64code)
    b = base64.b64decode(b64code)
    im = imread(io.BytesIO(b))
    print(im.shape)

    # check that the image shape is ready to be fed to the model
    if im.shape[:2] != (256, 256):
        return {"error": "not the right sizes"}

    if len(im.shape) < 3:
        im = np.expand_dims(im, axis=-1)

    if len(im.shape) == 3 and im.shape[2] > 1:
        im = np.mean(im, axis=-1)
        print(im.shape)

    image = tf.constant(im)

    assert app.state.model is not None

    # add a dimension for feeding to the model
    image = np.expand_dims(image, axis=0)

    # make sure the pixel color values are min-maxed
    image = (image / 255).astype(np.float32)

    # make prediction
    y_pred = app.state.model.predict(image)

    # get virus with max probability
    max = y_pred.argmax()
    label = VIRUSES[max]
    proba = y_pred[0][max]

    # return a json dictionnary with the predicted label and its probability
    return {"Virus": str(label), "Proba": round(float(proba), 2)}
