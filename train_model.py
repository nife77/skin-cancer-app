import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # increase later if you want better performance

def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_gen = train_datagen.flow_from_directory(
        'data/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        'data/val',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    print("Class indices:", train_gen.class_indices)
    return train_gen, val_gen

def build_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # ❗ Unfreeze the last ResNet50 block for fine-tuning
    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-10:]:
        layer.trainable = True

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def main():
    train_gen, val_gen = get_data_generators()
    model = build_model()
    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'skin_cancer_resnet50.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
