# Vehicle Fuel Efficiency Prediction Project
In this project, we are going to build an end-to-end ML workflow by using Vertex AI - Google Cloud's new managed ML platform. The steps taken in the project are listed below:
- Setup Working Environment.
- Building and containerizing model training code using Cloud Shell.
- Submitting a custom model training job to Vertex AI.
- Deploying the trained model to an endpoint, and use that endpoint to get predictions.

## 1. Setting-Up Working Environment:

Sign in to [Cloud Console](http://console.cloud.google.com/) and create a new project. Afterwards, start the [Cloud Shell](https://cloud.google.com/cloud-shell/) and run the following command in Cloud Shell to:

1. Confirm that you are authenticated:
```bash
gcloud auth list
```
2. Confirm that the gcloud command knows about your project:
```bash
gcloud config list project
```
3. Give your project access to the Compute Engine, Container Registry, and Vertex AI services:
```bash
gcloud services enable compute.googleapis.com         \
                       containerregistry.googleapis.com  \
                       aiplatform.googleapis.com
```
4. Create a bucket:
```bash
BUCKET_NAME=gs://$GOOGLE_CLOUD_PROJECT-bucket
gsutil mb -l us-central1 $BUCKET_NAME
```
5. Create an alias:
```bash
alias python=python3
```
## 2. Containerize Training Code:
We'll submit this training job to Vertex by putting our training code in a Docker container and pushing this container to [Google Container Registry]("https://cloud.google.com/container-registry?utm_campaign=CDR_sar_aiml_ucaiplabs_011321&utm_source=external&utm_medium=web). Now, in order to work in this task, run the following commands in your cloud shell to:

1. Create the files we will need for our Docker Container:
```bash
mkdir mpg
cd mpg
touch Dockerfile
mkdir trainer
touch trainer/train.py
```
2. Open the Dockerfile:
```bash
vim Dockerfile
```
3. Paste the below code in the Dockerfile
```bash
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3
WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]
```
Note: Press `ESC` key and then `:wq` to write and quit the vim editor.

3. Open the train.py file:
```bash
vim trainer/train.py
```
4. Paste the below code in the train.py file:
```bash
# This will be replaced with your bucket name after running the `sed` command in the tutorial
BUCKET = "BUCKET_NAME"

import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

"""## The Auto MPG dataset

The dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/).

### Get the data
First download the dataset.
"""

"""Import it using pandas"""

dataset_path = "https://storage.googleapis.com/io-vertex-codelab/auto-mpg.csv"
dataset = pd.read_csv(dataset_path, na_values = "?")

dataset.tail()

"""### Clean the data

The dataset contains a few unknown values.
"""

dataset.isna().sum()

"""To keep this initial tutorial simple drop those rows."""

dataset = dataset.dropna()

"""The `"origin"` column is really categorical, not numeric. So convert that to a one-hot:"""

dataset['origin'] = dataset['origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.tail()

"""### Split the data into train and test

Now split the dataset into a training set and a test set.

We will use the test set in the final evaluation of our model.
"""

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

"""### Inspect the data

Have a quick look at the joint distribution of a few pairs of columns from the training set.

Also look at the overall statistics:
"""

train_stats = train_dataset.describe()
train_stats.pop("mpg")
train_stats = train_stats.transpose()
train_stats

"""### Split features from labels

Separate the target value, or "label", from the features. This label is the value that you will train the model to predict.
"""

train_labels = train_dataset.pop('mpg')
test_labels = test_dataset.pop('mpg')

"""### Normalize the data

Look again at the `train_stats` block above and note how different the ranges of each feature are.

It is good practice to normalize features that use different scales and ranges. Although the model *might* converge without feature normalization, it makes training more difficult, and it makes the resulting model dependent on the choice of units used in the input.

Note: Although we intentionally generate these statistics from only the training dataset, these statistics will also be used to normalize the test dataset. We need to do that to project the test dataset into the same distribution that the model has been trained on.
"""

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

"""This normalized data is what we will use to train the model.

Caution: The statistics used to normalize the inputs here (mean and standard deviation) need to be applied to any other data that is fed to the model, along with the one-hot encoding that we did earlier.  That includes the test set as well as live data when the model is used in production.

## The model

### Build the model

Let's build our model. Here, we'll use a `Sequential` model with two densely connected hidden layers, and an output layer that returns a single, continuous value. The model building steps are wrapped in a function, `build_model`, since we'll create a second model, later on.
"""

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

"""### Inspect the model

Use the `.summary` method to print a simple description of the model
"""

model.summary()

"""Now try out the model. Take a batch of `10` examples from the training data and call `model.predict` on it.

It seems to be working, and it produces a result of the expected shape and type.

### Train the model

Train the model for 1000 epochs, and record the training and validation accuracy in the `history` object.

Visualize the model's training progress using the stats stored in the `history` object.

This graph shows little improvement, or even degradation in the validation error after about 100 epochs. Let's update the `model.fit` call to automatically stop training when the validation score doesn't improve. We'll use an *EarlyStopping callback* that tests a training condition for  every epoch. If a set amount of epochs elapses without showing improvement, then automatically stop the training.

You can learn more about this callback [here](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).
"""

model = build_model()

EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, 
                    callbacks=[early_stop])


# Export model and save to GCS
model.save(BUCKET + '/mpg/model')
```
Note: Again, press `ESC` key and then `:wq` to write and quit the vim editor.

5. Add your own bucket name to the file:
```bash
sed -i "s|BUCKET_NAME|$BUCKET_NAME|g" trainer/train.py
```
6. Define a variable with the URI of your container image in Google Container Registry: 
```bash
IMAGE_URI="gcr.io/$GOOGLE_CLOUD_PROJECT/mpg:v1"
```
7. Build the container:
```bash
docker build ./ -t $IMAGE_URI
```
8. push container to Google Container Registry:
```bash
docker push $IMAGE_URI
```
![Container-Image]("https://github.com/alihussainia/Vehicle-Fuel-Efficiency-Prediction/blob/main/images/container-image.png")
