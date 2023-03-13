from fastapi import FastAPI
from ddd.ml_logic.registry import load_model


app = FastAPI()
app.state.model = load_model()



@app.get('/')
def root():
    return {'youhou':'hourra'}



# def predict(image:tf):

#     model = load_model()
#     assert model is not None
#     image = np.expand_dims(image, axis=0)    #pour avoir le bon format
#     y_pred = model.predict(image)
#     max = y_pred.argmax()
#     label = CLASS_NAME[max]
#     print(label)
#     print(y_pred[0][max])
#     proba = y_pred[0][max]
#     return label, proba
