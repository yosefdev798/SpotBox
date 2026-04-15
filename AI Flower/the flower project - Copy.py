# =========================
# 1. Setup
# =========================
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
# =========================
# 2. Load Dataset
# =========================
dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)

train_ds = dataset['train']

# Split dataset
train_size = int(0.8 * info.splits['train'].num_examples)
val_size = int(0.1 * info.splits['train'].num_examples)
test_size = int(0.1 * info.splits['train'].num_examples)

val_ds = train_ds.skip(train_size).take(val_size)
test_ds = train_ds.skip(train_size + val_size).take(test_size)
train_ds = train_ds.take(train_size)

# =========================
# 3. Preprocessing
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32

def format_example(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    return image, label

train = train_ds.map(format_example).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val = val_ds.map(format_example).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test = test_ds.map(format_example).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =========================
# 4. Model
# =========================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

NUM_CLASSES = info.features['label'].num_classes

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# =========================
# 5. Compile
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 6. Train (Initial)
# =========================
history = model.fit(
    train,
    validation_data=val,
    epochs=5
)

# =========================
# 7. Fine-Tuning
# =========================
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train,
    validation_data=val,
    epochs=5
)

# =========================
# 8. Evaluate
# =========================
test_loss, test_acc = model.evaluate(test)
print("Test accuracy:", test_acc)


# Get label names
class_names = info.features['label'].names

# Take one batch from test dataset
for images, labels in test.take(1):
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    plt.figure(figsize=(10,10))

    for i in range(9):  # show 9 images
        plt.subplot(3,3,i+1)
        
        # convert image back (important for MobileNetV2 preprocessing)
        img = (images[i] + 1) / 2  # from [-1,1] to [0,1]

        plt.imshow(img)
        
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]

        plt.title(f"T: {true_label}\nP: {pred_label}")
        plt.axis("off")

    #plt.show()

# =========================
# 10. Save Model
# =========================
model.save("flower_model.keras")    