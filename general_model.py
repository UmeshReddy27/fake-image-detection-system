import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras import layers, models

def build_general_ai_detector(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False 
    
    # Hardwired inference mode
    x = base_model(inputs, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    model = build_general_ai_detector()
    model.summary()