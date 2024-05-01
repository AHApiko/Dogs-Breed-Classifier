import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

# Load the dataset from TensorFlow Datasets and print information about it
dataset, info = tfds.load('stanford_dogs', with_info=True)
print(info)
# Split the dataset into training and testing sets
train_dataset = dataset['train']
test_dataset = dataset['test']
# Define image data augmentation for training dataset
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.resnet.preprocess_input)
# Define a learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
# Define a function to preprocess each sample in the dataset
def preprocess_sample(sample):
    image = tf.image.resize(sample['image'], (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    label = sample['label']
    return image, label
# Define batch size and preprocess training and testing datasets
batch_size = 32
train_dataset = train_dataset.map(preprocess_sample).shuffle(buffer_size=1000).batch(batch_size)
test_dataset = test_dataset.map(preprocess_sample).batch(batch_size)
# Define image data augmentation for testing dataset
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)
# Load the Xception model with pre-trained weights
base_model = Xception(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
# Fine-tune the last 10 layers of the base model
for layer in base_model.layers[-10:]:
    layer.trainable = True
# Define the number of classes and build the classification head
num_classes = 120
x = Flatten()(base_model.output)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.7)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
# Train the model and validate on the test dataset
history = model.fit(train_dataset, validation_data=test_dataset, epochs=5)
# This are the results I get using googlecollabs GPU:
#Epoch 1/5
#375/375 [==============================] - 251s 563ms/step - loss: 2.9632 - accuracy: 0.4672 - val_loss: 1.5621 - val_accuracy: 0.7312
#Epoch 2/5
#375/375 [==============================] - 214s 566ms/step - loss: 1.2500 - accuracy: 0.7130 - val_loss: 1.1114 - val_accuracy: 0.7427
#Epoch 3/5
#375/375 [==============================] - 222s 588ms/step - loss: 0.8590 - accuracy: 0.7943 - val_loss: 1.1342 - val_accuracy: 0.7563
#Epoch 4/5
#375/375 [==============================] - 221s 586ms/step - loss: 0.5844 - accuracy: 0.8508 - val_loss: 1.2845 - val_accuracy: 0.7590
#Epoch 5/5
#375/375 [==============================] - 222s 588ms/step - loss: 0.4358 - accuracy: 0.8882 - val_loss: 1.3522 - val_accuracy: 0.7604

# Save the model, I used googlecollabs for computational power reasons so I saved it on google drive. 
#I will make a quite simple webpage app using flask to make it more interactive
from google.colab import drive
drive.mount('/content/drive')
model.save('/content/drive/MyDrive/my_model')

