import cv2
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('without_augmentation.keras')

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

        # Optionally save image
        # cv2.imwrite("signature_detected.jpg", frame)

        cv2.waitKey(0)  # Wait until a key is pressed before going back to live view

cap.release()
cv2.destroyAllWindows()
