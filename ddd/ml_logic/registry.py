import os
from ddd.params import *
import pickle
from tensorflow import keras
import mlflow
import time


def save_result(params: dict, metrics: dict) -> None:
    '''
    Save result model (parameters and metrics) locally
    '''

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    #save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params",
                                   timestamp + '.pickle')
        with open(params_path, 'wb') as file:
            pickle.dump(params, file)

    # save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics",
                                    timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model: keras.Model = None) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print('✅ Model saved locally')

    if MODEL_TARGET == 'mlflow':
        mlflow.tensorflow.log_model(model=model,
                                    artifact_path='model',
                                    registered_model_name=MLFLOW_MODEL_NAME)
        print("✅ Model saved to mlflow")

        return None

    return None


#def load_model(stage='Production') -> keras.Model:
# pour l'instant on s'en fou


def mlflow_run(func):
    """Generic function to log params and results to mlflow along with tensorflow autologging

    Args:
        func (function): Function you want to run within mlflow run
        params (dict, optional): Params to add to the run in mlflow. Defaults to None.
        context (str, optional): Param describing the context of the run. Defaults to "Train".
    """

    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)
        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)
        print("✅ mlflow_run autolog done")
        return results

    return wrapper
