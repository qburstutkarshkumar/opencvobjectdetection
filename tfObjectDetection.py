import cv2  # OpenCV

# Caffe: Requires .prototxt and .caffemodel files.
# TensorFlow: Requires .pb (frozen graph) and optionally a .pbtxt file.
# ONNX: Requires an .onnx file.

#https://github.com/ankityddv/ObjectDetector-OpenCV/tree/main
#https://github.com/ankityddv/ObjectDetector-OpenCV/blob/main/frozen_inference_graph.pb
#https://github.com/ankityddv/ObjectDetector-OpenCV/blob/main/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt

configfile = "/home/utkarshkumar/Projects/Object_Detection/dnnDetection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Config File for Trained MobileNet SSD Model
frozen_model = "/home/utkarshkumar/Projects/Object_Detection/dnnDetection/frozen_inference_graph.pb"  # Weighhts - derived from tensor flow

model = cv2.dnn_DetectionModel(frozen_model, configfile)  # Model used to Detect Objects

Class_labels = []  # 91 Classes of Classification available in the COCO Dataset Collection

with open("/home/utkarshkumar/Projects/Object_Detection/dnnDetection/coco.names",'rt') as f:
    Class_labels = f.read().rstrip('\n').split('\n')
# Customizing input format provided to our Model
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# For Webcam
cap = cv2.VideoCapture("/home/utkarshkumar/Projects/Object_Detection/4K Road traffic video for object detection and tracking - free download now!.mp4")

# Set the desired frame width and height
width = 1280  # Desired width
height = 720  # Desired height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Font options used in labelling the Objects in Frame
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    if ret == True:
        ClassIndex, Confidence, bbox = model.detect(frame, confThreshold=0.5)  # Class Index values start from 1 
        if len(ClassIndex) != 0:  # if Objects Found
            for ClassInd, Confid, Boxes in zip(ClassIndex.flatten(), Confidence.flatten(), bbox):
                if ClassInd >=1 and ClassInd <=80:
                    cv2.rectangle(frame, Boxes, (255, 0, 0), 2)  # Boxes overlayed around the Detected object
                    string = str(Class_labels[ClassInd-1]).title()+" "+format(float(Confid),".2f")  # the Label Displayed in the Image
                    cv2.putText(frame, string, (Boxes[0]+10, Boxes[1]-10), font, fontScale=font_scale, color=(0,255,255), thickness=2)
        cv2.imshow("OBJECT DETECTION", frame)
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()