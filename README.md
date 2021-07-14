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
## 2. Containerizing Training Code:
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

4. Open the train.py file:
```bash
vim trainer/train.py
```
5. Paste the below code in the train.py file:
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

6. Add your own bucket name to the file:
```bash
sed -i "s|BUCKET_NAME|$BUCKET_NAME|g" trainer/train.py
```
7. Define a variable with the URI of your container image in Google Container Registry: 
```bash
IMAGE_URI="gcr.io/$GOOGLE_CLOUD_PROJECT/mpg:v1"
```
8. Build the container:
```bash
docker build ./ -t $IMAGE_URI
```
9. push container to Google Container Registry:
```bash
docker push $IMAGE_URI
```
To verify your image was pushed to Container Registry, you should see `mpg` folder when you navigate to the [Container Registry](https://console.cloud.google.com/gcr) section of your console:

![Container-Image](https://github.com/alihussainia/Vehicle-Fuel-Efficiency-Prediction/blob/main/images/container-image.png)

## 3. Running a training job on Vertex AI 

In this project, we are using custom training via our own custom container on Google Container Registry. Please follow the step by step guide mentioned on this page [link](https://codelabs.developers.google.com/codelabs/vertex-ai-custom-models#4) to submit a custom model training job to Vertex AI.

After successfully submitting the training job, you will see something like that in your Cloud Console:

![Training-Image](https://github.com/alihussainia/Vehicle-Fuel-Efficiency-Prediction/blob/main/images/training.png)

## 4. Deploying a model endpoint:

In this step we'll create an endpoint for our trained model. We can use this to get predictions on our model via the Vertex AI API. Run the following commands in your Cloud Shell to:

1. Install the Vertex AI SDK:
```bash
pip3 install google-cloud-aiplatform --upgrade --user
```
2. Open the `deploy.py` file:
```bash
vim deploy.py
```
3. Paste the below code in the deploy.py file:
```bash
from google.cloud import aiplatform

# Create a model resource from public model assets
model = aiplatform.Model.upload(
    display_name="mpg-imported",
    artifact_uri="gs://io-vertex-codelab/mpg-model/",
    serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-3:latest"
)

# Deploy the above model to an endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4"
)
```
Note: Again, press `ESC` key and then `:wq` to write and quit the vim editor.

4. Switch back into your root dir, and run this deploy.py file:
```bash
cd ..
python3 deploy.py | tee deploy-output.txt
```
Note:
This will take 10-15 minutes to run. To ensure it's working correctly, navigate to the `Models` section of your console in Vertex AI:

![Model](https://github.com/alihussainia/Vehicle-Fuel-Efficiency-Prediction/blob/main/images/model.png)

Click on `mgp-imported` and you should see your endpoint for that model being created:

![Endpoint-Image](https://github.com/alihussainia/Vehicle-Fuel-Efficiency-Prediction/blob/main/images/endpoint.png)

In your Cloud Shell Terminal, you'll see something like the following image:

![deploy-shell-image](https://github.com/alihussainia/Vehicle-Fuel-Efficiency-Prediction/blob/main/images/deploy.png)

5. Open the `predict.py` file:
```bash
vim predict.py
```
6. Paste the below code in the predict.py file:
```bash
from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="ENDPOINT_STRING"
)

# A test example we'll send to our model for prediction
test_mpg = [1, 2, 3, 2, -2, -1, -2, -1, 0]

response = endpoint.predict([test_mpg])

print('API response: ', response)

print('Predicted MPG: ', response.predictions[0][0])
```
Note: Again, press `ESC` key and then `:wq` to write and quit the vim editor.

7. Replace `ENDPOINT_STRING` in the predict.py file with your own endpoint:
```bash
ENDPOINT=$(cat deploy-output.txt | sed -nre 's:.*Resource name\: (.*):\1:p' | tail -1)
sed -i "s|ENDPOINT_STRING|$ENDPOINT|g" predict.py
```
8. Run the `predict.py` file to get a prediction from our deployed model endpoint:
```bash
python3 predict.py
```
Once, you will run the above command, you will see this kind of result in your Cloud Shell:

![Prediction-Image](https://github.com/alihussainia/Vehicle-Fuel-Efficiency-Prediction/blob/main/images/predict.png)

## Dataset:
In this project, the [Auto-mpg dataset](https://www.kaggle.com/uciml/autompg-dataset) has been used. The data is technical spec of cars and is also available in  [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/auto+mpg) 

## References:
[Build and deploy a model with Vertex AI - CodeLabs](https://codelabs.developers.google.com/codelabs/vertex-ai-custom-models#0)



































