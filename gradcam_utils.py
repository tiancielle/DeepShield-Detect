
import numpy as np
import cv2
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras

GRADCAM_LAYER = "out_relu"  # MobileNetV2

def get_gradcam_heatmap(model, img_array, layer_name=GRADCAM_LAYER):
    mobilenet_model = None
    for layer in model.layers:
        if hasattr(layer, "layers"):
            mobilenet_model = layer
            break
    if mobilenet_model is None:
        raise ValueError("Sous-modèle MobileNetV2 introuvable.")
    layer_names = [l.name for l in mobilenet_model.layers]
    if layer_name not in layer_names:
        candidates = [l.name for l in mobilenet_model.layers
                      if "relu" in l.name or "activation" in l.name]
        layer_name = candidates[-1]
    grad_model = keras.Model(
        inputs=mobilenet_model.input,
        outputs=[mobilenet_model.get_layer(layer_name).output,
                 mobilenet_model.output]
    )
    with tf.GradientTape() as tape:
        img_tensor          = tf.cast(img_array, tf.float32)
        conv_outputs, preds = grad_model(img_tensor, training=False)
        loss                = preds[:, 0]
    grads        = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_array, heatmap, alpha=0.4):
    h_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    h_colored = cm.jet(h_resized)[:, :, :3]
    result    = (1 - alpha) * img_array.astype("float32") + alpha * h_colored
    return np.clip(result, 0, 1)
