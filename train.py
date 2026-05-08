import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from face_model import build_deepfake_detector

DATA_DIR = "dataset" 
BATCH_SIZE = 32      
IMG_SIZE = (224, 224)
EPOCHS = 20 # (Change to 1 for a fast test!)         

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomZoom(0.1),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

model = build_deepfake_detector(input_shape=(224, 224, 3))

callbacks = [
    EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
    # WEIGHTS ONLY SAVE
    ModelCheckpoint('best_deepfake_model.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)
print("Training Complete! Memories saved.")