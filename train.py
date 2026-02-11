import tensorflow as tf
from tensorflow.keras import layers,models
import os
import warnings
warnings.filterwarnings('ignore')

try:
    # find path
    base_dir = 'data'
    train_dir=os.path.join(base_dir, 'train')

    # Set parameter of image
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32

    print("------------------Loading the Images----------------")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Categores detected: {class_names}")

    # Create models(brain)
    model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)), # Newer way to define input
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
    ])

    print("Model structure is ready! ")
    model.summary()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # Start the training
    print("-----------------------------Traning Starting---------------------------------")
    epochs=10
    history=model.fit(
        train_ds,
        epochs = epochs
    )


    # Save the brain
    model.save("models/my_first_model.keras")
    print("Model saved in models/Folder!")

except Exception as e:
    print("Error: ",e)
