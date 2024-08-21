import cv2
import numpy as np

# https://github.com/chuanqi305/MobileNet-SSD
# https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel
# https://github.com/chuanqi305/MobileNet-SSD/blob/master/deploy.prototxt

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe('/home/utkarshkumar/Projects/Object_Detection/caffemodel/deploy.prototxt', '/home/utkarshkumar/Projects/Object_Detection/caffemodel/mobilenet_iter_73000.caffemodel')
# For Webcam
cap = cv2.VideoCapture(0)

# Set the desired frame width and height
width = 1480  # Desired width
height = 720  # Desired height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Font options used in labelling the Objects in Frame
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

# Caffe: Requires .prototxt and .caffemodel files.
# TensorFlow: Requires .pb (frozen graph) and optionally a .pbtxt file.
# ONNX: Requires an .onnx file.


class_labels = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"]

while True:
    ret, image = cap.read()

    # Load an image
    (h, w) = image.shape[:2]

    # Prepare the image as input
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    # Perform forward pass to get detections
    detections = net.forward()
    # Draw detections on the image
    #[batch_id, class_id, confidence, x_min, y_min, x_max, y_max]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Filter by confidence threshold
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{class_labels[idx]},{confidence:.2f}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("OBJECT DETECTION", image)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()