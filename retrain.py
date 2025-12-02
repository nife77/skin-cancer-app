import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

# Load ResNet50 base
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

# Freeze all layers except last block
for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-30:]:  # Unfreeze last conv block
    layer.trainable = True

# Build classifier head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Data pipeline
datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE)
val_gen = datagen.flow_from_directory(VAL_DIR, target_size=IMAGE_SIZE)

# Train for 3–5 epochs
model.fit(train_gen, validation_data=val_gen, epochs=3)

# Save FULL model including conv layers
model.save("models/skin_cancer_resnet50_FIXED.keras")
    