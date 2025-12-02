# app/gradcam_utils.py
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
import PIL.Image as Image

def find_last_conv_layer(model):
    """Return name of last 4D convolutional layer in model (robust to nested models)."""
    from tensorflow.keras.models import Model as KModel
    # iterate reversed layers to find conv
    for layer in reversed(model.layers):
        # if layer is a nested model, search inside
        if isinstance(layer, KModel):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                continue
        # check class or instance
        try:
            if isinstance(layer, Conv2D) or layer.__class__.__name__.lower().startswith('conv'):
                # check output shape is 4D (batch, H, W, C)
                out_shape = getattr(layer, "output_shape", None)
                if out_shape is not None and len(out_shape) == 4:
                    return layer.name
        except Exception:
            # ignore layers without output_shape
            continue
    raise ValueError("No convolutional (4D) layer found in model.")

def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None, pred_index=None):
    """
    Compute Grad-CAM heatmap (2D array normalized 0..1).
    img_array: preprocessed batch (1, H, W, 3) - same preprocessing used for prediction.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = Model(inputs=model.inputs,
                       outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    # If grads is None for some reason, return zeros heatmap sized to conv output
    if grads is None:
        conv_shape = conv_outputs.shape
        h = int(conv_shape[1])
        w = int(conv_shape[2])
        return np.zeros((h, w), dtype=np.float32)

    # global-average-pool grads -> channel weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

    conv_outputs = conv_outputs[0].numpy()  # (h, w, channels)

    # weight conv channels by gradients
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.sum(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    # normalize to 0..1
    maxv = np.max(heatmap)
    if maxv == 0 or np.isnan(maxv):
        return np.zeros_like(heatmap, dtype=np.float32)
    heatmap = heatmap / maxv
    heatmap = heatmap.astype(np.float32)
    return heatmap

def overlay_heatmap_on_image(pil_image, heatmap, alpha=0.4):
    """
    Overlay heatmap on top of a PIL.Image and return an RGB uint8 numpy array.
    - pil_image: PIL.Image (original full-size)
    - heatmap: 2D numpy array (values 0..1) — small (e.g., 7x7) or larger
    - alpha: blending weight for heatmap (0..1)
    """
    if not isinstance(pil_image, Image.Image):
        raise ValueError("pil_image must be a PIL.Image")

    img = np.array(pil_image.convert('RGB')).astype(np.float32)  # shape (H, W, 3)
    H, W = img.shape[0], img.shape[1]

    # resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))

    # apply color map (returns BGR)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    # convert to RGB
    heatmap_color = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    img_norm = img / 255.0

    # blend (pixelwise) — ensure same shape
    if heatmap_color.shape[:2] != img_norm.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (img_norm.shape[1], img_norm.shape[0]))

    superimposed = (1.0 - alpha) * img_norm + alpha * heatmap_color
    superimposed = np.clip(superimposed * 255.0, 0, 255).astype(np.uint8)
    return superimposed
