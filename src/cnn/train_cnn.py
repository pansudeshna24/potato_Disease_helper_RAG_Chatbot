# src/cnn/train_cnn.py

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# =====================================================

# CONFIG

# =====================================================

DATA_PATH = "data/image_reference_index"

MODEL_SAVE_PATH = (
"src/cnn/models/potato_efficientnet.keras"
)

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42

AUTOTUNE = tf.data.AUTOTUNE

# =====================================================

# LOAD DATASET

# =====================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
DATA_PATH,
validation_split=0.2,
subset="training",
seed=SEED,
image_size=IMG_SIZE,
batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
DATA_PATH,
validation_split=0.2,
subset="validation",
seed=SEED,
image_size=IMG_SIZE,
batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

NUM_CLASSES = len(class_names)

print("\nClasses:")
print(class_names)

# =====================================================

# CLASS WEIGHTS

# =====================================================

y_train = np.concatenate(
[y for _, y in train_ds],
axis=0
)

class_weights = compute_class_weight(
class_weight="balanced",
classes=np.unique(y_train),
y=y_train
)

class_weights = dict(
enumerate(class_weights)
)

print("\nClass Weights:")
print(class_weights)

# =====================================================

# PREPROCESSING

# =====================================================

def preprocess(image, label):

```
image = (
    tf.keras.applications.efficientnet
    .preprocess_input(image)
)

return image, label
```

train_ds = (
train_ds
.map(preprocess, num_parallel_calls=AUTOTUNE)
.cache()
.shuffle(1000)
.prefetch(AUTOTUNE)
)

val_ds = (
val_ds
.map(preprocess, num_parallel_calls=AUTOTUNE)
.cache()
.prefetch(AUTOTUNE)
)

# =====================================================

# DATA AUGMENTATION

# =====================================================

data_augmentation = tf.keras.Sequential([

```
tf.keras.layers.RandomFlip("horizontal"),

tf.keras.layers.RandomRotation(0.1),

tf.keras.layers.RandomZoom(0.1),

tf.keras.layers.RandomContrast(0.1),
```

])

# =====================================================

# BASE MODEL

# =====================================================

base_model = (
tf.keras.applications.EfficientNetB0(
include_top=False,
weights="imagenet",
input_shape=(224, 224, 3)
)
)

base_model.trainable = False

# =====================================================

# BUILD MODEL

# =====================================================

inputs = tf.keras.Input(
shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
x,
training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(
256,
activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.5)(x)

outputs = tf.keras.layers.Dense(
NUM_CLASSES,
activation="softmax"
)(x)

model = tf.keras.Model(
inputs,
outputs
)

# =====================================================

# COMPILE

# =====================================================

model.compile(

```
optimizer=tf.keras.optimizers.Adam(
    1e-3
),

loss="sparse_categorical_crossentropy",

metrics=["accuracy"]
```

)

model.summary()

# =====================================================

# CALLBACKS

# =====================================================

callbacks = [

```
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
),

tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    verbose=1
)
```

]

# =====================================================

# TRAIN FEATURE EXTRACTION

# =====================================================

print("\nStarting Training...\n")

model.fit(

```
train_ds,

validation_data=val_ds,

epochs=20,

class_weight=class_weights,

callbacks=callbacks
```

)

# =====================================================

# FINE TUNING

# =====================================================

print("\nStarting Fine Tuning...\n")

base_model.trainable = True

for layer in base_model.layers[:-20]:
layer.trainable = False

model.compile(

```
optimizer=tf.keras.optimizers.Adam(
    1e-4
),

loss="sparse_categorical_crossentropy",

metrics=["accuracy"]
```

)

model.fit(

```
train_ds,

validation_data=val_ds,

epochs=10,

class_weight=class_weights,

callbacks=callbacks
```

)

# =====================================================

# EVALUATION

# =====================================================

y_true = np.concatenate(
[y for _, y in val_ds],
axis=0
)

y_pred = np.argmax(
model.predict(val_ds),
axis=1
)

print("\nClassification Report:\n")

print(
classification_report(
y_true,
y_pred,
target_names=class_names,
zero_division=0
)
)

# =====================================================

# SAVE MODEL

# =====================================================

os.makedirs(
"src/cnn/models",
exist_ok=True
)

model.save(MODEL_SAVE_PATH)

print(
f"\nModel saved to: "
f"{MODEL_SAVE_PATH}"
)

print("\nTraining completed successfully.")
