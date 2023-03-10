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
print('model ready')


@app.get('/')
def root():
    return {'youhou':'hourra'}


@app.get('/predict/')
def root():
    return {'youhou':'hourra'}


class Item(BaseModel):
    image:str


@app.post('/predict/')
def predict(post_dict: Item):
    b64code = post_dict.image
    # print(b64code)
    b = base64.b64decode(b64code)
    im = imread(io.BytesIO(b))
    print(im.shape)
    # im2 = im[:,:,1]
    im2 = np.expand_dims(im, axis=2)
    image = tf.constant(im2)

    assert app.state.model is not None

    image = np.expand_dims(image, axis=0)    #pour avoir le bon format
    y_pred = app.state.model.predict(image)
    #récupérer le label avec le plus de probabilité
    max = y_pred.argmax()
    label = CLASS_NAME[max]
    print(label)
    print(y_pred[0][max])
    proba = y_pred[0][max]
    return {
        'Virus': str(label),
        'Proba = ': round(float(proba),2)
    }
