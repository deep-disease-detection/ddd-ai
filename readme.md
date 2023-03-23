# Deep Disease Detection Model

This package contains all the code required to train, evaluate and predict virus classes using our different CNN models.


## Setup
1. (Optional) Make sure you set-up a Python virtual environment
2. Make sure you have a complete .env file. You can use the .env.sample file to check which variables need to be set
3. Install the package using pip
4. Make sure you <a href="https://data.mendeley.com/datasets/kxsvzhcfgs/2"> download the dataset </a> and unzip it in the **data** folder. Double check that all path environment variables are set to point to the right folders.

## Repository structure
### The DDD package
#### API
- **fast**: Defines the API endpoints to use the classification model of choice to predict the virus type of an image.
- **post_image**: Script to test the API.

#### Interface
- **main**: main package method including a method to get preprocessed Tf.Dataset objects for training, a method to train the model and to evaluate it as well as a method to make a prediction.

### ml_logic
- **models**: contains methods to intialize and train the different models available in the package. The models are: Custom CNN from the paper, DenseNet201, VGG19 as well as two dummy models (CNN and test).
- **preprocess**: contains all methods to preprocess the raw dataset. Check out the preprocessing section on this readme for more information.
- **registry**: Methods to load and save model weights either locally, on MLFlow or on a Google Cloud Bucket. Includes methods to save results and training parameters.

### Other
- **params**: All package parameters. Some are loaded from .env file while others are set in this file.
- **utils**: Helper methods.

## Data Preprocessing

### Image processing
For our classification task, the goal of the data preprocessing pipeline is to:
- Generate 256x256 px images of each virus particle on the images with the same resolution
- Augment the data to have a balanced training set


Data preprocessing involved several key steps.

- **Loading and formatting** the data from the image folders and the annotations : virus particle positions and class
- **Resize** the image: Microscope images have different resolutions (nm per pixel). Using the meta-data of the picture, we make sure to rescale the X and Y axis of the image to have the same resolution accross all images.
- **Min-max scale** the image: The pixel values were not normalized, so we made sure they were integers between 0 and 255 to be able to save the pictures as .png later on.
- Add **padding**: Add 256px padding on all sides of the image to avoid potential issues when cropping the viruses. We used mirror padding for this.
- **Crop** around the viruses: Using the meta-data collected (position of each virus particle on each image), generate 256x256px images around each virus.
- **Save the images** in a folder.

For all of the above tasks, we used the python opencv package.

### Image augmentation
🚧 Work in Progress 🚧

## Models
We tried several architectures that were proposed in the [scientific paper](https://www.sciencedirect.com/science/article/pii/S0169260721003928)

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0169260721003928-gr2.jpg"> &nbsp;
*Source: Damian J. Matuszewski, Ida-Maria Sintorn,
TEM virus images: Benchmark dataset and deep learning classification*

### Custom CNN architecture
We implemented the custom CNN architecture proposed in the scientific paper using Tensorflow.

We obtained satisfying accuracy (around 80%) after 70 epochs of training.

### DenseNet (tranfer learning)
We used transfer learning on the DenseNet201 architecture pre-trained on ImageNet.

The accuracy after around 30 epochs was 84%.
