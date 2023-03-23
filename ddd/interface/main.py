from ddd.params import *
from ddd.ml_logic.models import *
from ddd.ml_logic.registry import save_model, save_result
import os
from ddd.ml_logic.registry import mlflow_run
import numpy as np
from ddd.ml_logic.registry import load_model
from tensorflow.keras import Model
from ddd.ml_logic.preprocess import convert_b64_to_tf
import cv2
import base64

# Dictionnary to select the right methods depending on the model choice of the user
MODEL_METHODS = {
    "custom": {
        "init": initialize_custom_model,
        "train": train_custom_model_fromdataset,
    },
    "dense": {
        "init": initialize_DenseNet_model,
        "train": train_DenseNet_model_fromdataset,
    },
    "vgg19": {"init": VGG19_model, "train": train_VGG19_model_fromdataset},
    "cnn": {"init": initialize_CNN_model, "train": train_CNN_model},
    "test": {"init": test_model, "train": train_test_model},
}


def get_dataset(normalize: bool = False) -> tuple:
    """
    Return processed train, validation and test Tf.Dataset objects
    """

    from tensorflow.keras.utils import image_dataset_from_directory

    print("Getting train dataset :)")
    # pour prendre un sample de train dataset
    if TEST == "True":
        train = image_dataset_from_directory(
            SAMPLE_PATH,
            labels="inferred",
            label_mode="categorical",
            shuffle=True,
            seed=42,
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
        )
    else:
        train = image_dataset_from_directory(
            AUGTRAIN_PATH,
            labels="inferred",
            label_mode="categorical",
            shuffle=True,
            seed=42,
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
        )

    print("Getting validation dataset :)")
    validation = image_dataset_from_directory(
        VALIDATION_PATH,
        labels="inferred",
        label_mode="categorical",
        shuffle=True,
        seed=42,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
    )

    print("Getting test dataset :)")
    test = image_dataset_from_directory(
        TEST_PATH,
        labels="inferred",
        label_mode="categorical",
        shuffle=True,
        seed=42,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
    )

    print("Min maxing data...")
    train = train.map(lambda x, y: (tf.cast(x / 255, tf.float32), y))
    validation = validation.map(lambda x, y: (tf.cast(x / 255, tf.float32), y))
    test = test.map(lambda x, y: (tf.cast(x / 255, tf.float32), y))

    if normalize:
        print("Normalizing data...")
        mean = np.zeros((256, 256, 1))
        total = 0

        for i in train.as_numpy_iterator():
            total += i[0].shape[0]
            mean += np.sum(i[0], axis=0)

        mean = mean / total

        var = np.zeros((256, 256, 1))
        total = 0

        for i in train.as_numpy_iterator():
            var_mean = np.repeat(np.expand_dims(mean, axis=0), i[0].shape[0], axis=0)
            var += np.sum((i[0] - var_mean) ** 2, axis=0)
            total += i[0].shape[0]

        std = np.sqrt(var / total)

        train = train.map(lambda x, y: ((x - mean) / std, y))
        validation = validation.map(lambda x, y: ((x - mean) / std, y))
        test = test.map(lambda x, y: ((x - mean) / std, y))

        print("Saving mean and std for predict")

        mean_path = os.path.join(LOCAL_REGISTRY_PATH, "state", "mean.pickle")

        std_path = os.path.join(LOCAL_REGISTRY_PATH, "state", "std.pickle")

        with open(mean_path, "wb") as file:
            np.save(file, mean)

        with open(std_path, "wb") as file:
            np.save(file, std)

    print("All done !✅")
    return train, validation, test


@mlflow_run
def train_model(choice_model: str = "custom"):

    # get all datasets
    train, val, test = get_dataset()

    model = MODEL_METHODS.get(choice_model).get("init")()

    model, history = MODEL_METHODS.get(choice_model).get("train")(model, train, val)

    val_accuracy = np.max(history.history.get("val_categorical_accuracy"))

    metrics = {
        "history": history,
        "val_accuracy": val_accuracy,
    }

    params = {
        "model_type": choice_model,
        "context": "train",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "augmentation": AUGMENTED,
    }

    save_result(params=params, metrics=metrics)
    save_model(model)

    return metrics.get("val_accuracy")


def evaluate_model(choice_model) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return ...
    """
    train, val, test = get_dataset()
    model = load_model()
    assert model is not None

    if choice_model in ["dense", "vgg19"]:
        test = test.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))

    metrics_dict = model.evaluate(test, verbose=0, return_dict=True)

    loss = metrics_dict["loss"]
    accuracy = metrics_dict["categorical_accuracy"]

    params = dict(context="evaluate")

    print("The model loss is ", loss)
    print("The model accuracy is ", accuracy)

    save_result(params=params, metrics=metrics_dict)
    return accuracy


def predict(image: np.array):

    model = load_model()
    assert model is not None
    image = np.expand_dims(image, axis=0)  # pour avoir le bon format

    # Min Maxing the image
    image = (image / 255).astype(np.float32)

    y_pred = model.predict(image)
    max = y_pred.argmax()
    label = VIRUSES[max]
    print(label)
    print(y_pred[0][max])
    proba = y_pred[0][max]
    return label, proba


if __name__ == "__main__":
    image = cv2.imread(
        "data/dataset-processed/TEM virus dataset/context_virus_1nm_256x256/augmented_train/Adenovirus/A4-65k-071120_2_0.png"
    )
    image2 = cv2.imencode(".png", image)[1]
    b64 = base64.b64encode(image2)
    X = convert_b64_to_tf(b64)
    predict(X)
