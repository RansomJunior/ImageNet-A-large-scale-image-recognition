# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:56:02 2023

@author: Tafadzwa Ransom Junior Mheuka
"""

import tensorflow as tf

# Define dataset parameters
batch_size = 32
image_size = (224, 224)

# Load ImageNet dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/path/to/imagenet/train',
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=123
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/path/to/imagenet/train',
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=123
)

# Preprocess images
train_ds = train_ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y))

# Define ResNet50 model
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(image_size[0], image_size[1], 3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add new classification layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax')(x)

# Create model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Train model
model.fit(train_ds, epochs=10, validation_data=val_ds)

'''
This code loads the ImageNet dataset using the tf.keras.preprocessing.image_dataset_from_directory() function, 
which reads images from a directory on disk and preprocesses them by resizing and normalizing them.
 It then uses the ResNet50 architecture, a popular deep neural network for image classification, as the base model, and adds new classification layers on top. 
 The model is compiled with the Adam optimizer and the Sparse Categorical Crossentropy loss function, and trained for 10 epochs.
'''


import tensorflow_datasets as tfds

# Load the ImageNet dataset
dataset, info = tfds.load('imagenet2012', split='train', with_info=True)

# Print dataset information
print(info)

# Iterate over the dataset
for example in dataset.take(10):
    # Extract the image and label
    image, label = example['image'], example['label']
    # Do something with the image and label, such as train a deep neural network for image classification



'''
This code uses the TensorFlow Datasets library to load the ImageNet dataset,
 and then iterates over the dataset to extract the images and labels. 
 You can use these images and labels to train a deep neural network for image classification,
 using techniques such as convolutional neural networks (CNNs).
 '''