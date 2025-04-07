import cv2
import tensorflow as tf
import numpy as np

def iou_loss(y_true, y_pred):
    cx_true, cy_true, w_true, h_true = tf.split(y_true, 4, axis=-1)
    cx_pred, cy_pred, w_pred, h_pred = tf.split(y_pred, 4, axis=-1)

    x1_true = cx_true - w_true / 2.0
    y1_true = cy_true - h_true / 2.0
    x2_true = cx_true + w_true / 2.0
    y2_true = cy_true + h_true / 2.0

    x1_pred = cx_pred - w_pred / 2.0
    y1_pred = cy_pred - h_pred / 2.0
    x2_pred = cx_pred + w_pred / 2.0
    y2_pred = cy_pred + h_pred / 2.0

    x1_inter = tf.maximum(x1_true, x1_pred)
    y1_inter = tf.maximum(y1_true, y1_pred)
    x2_inter = tf.minimum(x2_true, x2_pred)
    y2_inter = tf.minimum(y2_true, y2_pred)

    inter_area = tf.maximum(0.0, x2_inter - x1_inter) * tf.maximum(0.0, y2_inter - y1_inter)

    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    union_area = true_area + pred_area - inter_area


    iou = inter_area / (union_area + 1e-6)

    # IoU loss = 1 - IoU
    return 1.0 - tf.squeeze(iou, axis=-1)

# Load trained model
model = tf.keras.models.load_model('cnn_signature_model_v1.keras')

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open video device")
    exit()

print("Press SPACE to take a picture and detect signature")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Live Preview - Press SPACE to Capture", frame)

    key = cv2.waitKey(1)

    if key % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif key % 256 == 32:
        # SPACE pressed – capture and process
        original_height, original_width = frame.shape[:2]

        # Preprocess
        input_frame = cv2.resize(frame, (224, 224))
        input_tensor = np.expand_dims(input_frame / 255.0, axis=0)

        # Predict
        prediction = model.predict(input_tensor)[0]
        cx, cy, w, h = prediction  # normalized center x, y, width, height

        # Convert to pixel values
        cx *= original_width
        cy *= original_height
        w *= original_width
        h *= original_height

        x_min = int(cx - w / 2)
        y_min = int(cy - h / 2)
        x_max = int(cx + w / 2)
        y_max = int(cy + h / 2)

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, "Signature", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show result
        cv2.imshow("Detected Signature", frame)

        cv2.waitKey(0)  # Wait until a key is pressed before going back to live view

cap.release()
cv2.destroyAllWindows()
