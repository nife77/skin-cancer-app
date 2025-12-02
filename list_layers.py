import tensorflow as tf

model = tf.keras.models.load_model("models/skin_cancer_resnet50.h5")

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output.shape if hasattr(layer, "output") else None)
import tensorflow as tf

model = tf.keras.models.load_model("models/skin_cancer_resnet50.h5")

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output.shape if hasattr(layer, "output") else None)
