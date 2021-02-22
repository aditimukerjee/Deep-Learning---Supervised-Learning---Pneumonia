# Deep-Learning---Supervised-Learning---Pneumonia

Here I am going to implement convolutional neural network (CNN) from scratch in Keras. This implement has been performed to identify whether a person has pneumonia or not.

You can download the dataset from the link below.
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

LOADING THE DATASET AND SPLITTING IT INTO TEST, TRAIN AND VALIDATION SETS

Once you have loaded the dataset in google colab I then moved on with loading the libraries that will be needed to implement the CNN. Here I used the Sequential method as I am creating a sequential model.

DATA AUGMENTATION

Here I have imported ImageDataGenerator from keras.preprocessing. The objective of ImageDataGenerator is to import data with labels easily into the model. It is a very useful class as it has many function to rescale, rotate, zoom, flip etc. The most useful thing about this class is that it doesn't affect the data stored on the disk. This class alters the data on the go while passing it to the model.

DEFINING THE MODEL

As it was my first time building a CNN model by myself, I built a very basic model.
Initially, there are one convolutional layers followed by a max-pooling layer to pick important features from the convolution matrix and then a dropout layer
Flatten Layer flattens n-dimension matrix into 1-D so that it could be passed into Dense Layers
Two Dense Layers which are fully connected: first layers have 128 neurons and another has 1 neuron to give results which would be binary neural network.

MODEL SUMMARY

You can check the summary of the model which I created. Here we have about 3.7 million parameters to train.

DEFINING THE CALLBACKS

After the creation of the model, I will import ModelCheckpoint and EarlyStopping method from Keras. I will create an object of both and pass that as callback functions to fit_generator.

ModelCheckpoint helps us to save the model by monitoring a specific parameter of the model. In this case, I am monitoring validation accuracy by passing val_acc to ModelCheckpoint. The model will only be saved to disk if the validation accuracy of the model in the current epoch is greater than what it was in the last epoch.

EarlyStopping helps us to stop the training of the model early if there is no increase in the parameter which I have set to monitor in EarlyStopping. In this case, I am monitoring validation accuracy by passing val_accuracy to EarlyStopping. I have here set patience to 5 which means that the model will stop to train if it doesn't see any rise in validation accuracy in 5 epochs.

I have defined the third callback. Here we defined a function that will reduce the learning rate by half if for every 10 epoch. This will help us in reaching the global minimum faster in the beginning and trying to converge at it once we reach there.

TRAINING THE MODEL

I am using model.fit_generator as I am using ImageDataGenerator to pass data to the model. I will pass train and validation data to fit_generator. In fit_generator steps_per_epoch will set the batch size to pass training data to the model and validation_steps will do the same for test data. You can tweak it based on your system specifications.
After executing the above line the model will start to train and you will start to see the training/validation accuracy and loss.

TESTING THE MODEL AND EXTRACTING THE METRICS

Now is the moment of truth:
Let's test the model and see its performance.
Here we have an accuracy of 63% and a loss of 0.69
