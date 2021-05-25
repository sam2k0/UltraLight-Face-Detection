import cv2
import dlib
import numpy as np
from imutils import face_utils

import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare

from box_utils import *
 
onnx_model = onnx.load('UltraLight-Face-Detector/ultra_light_640.onnx') 
predictor = prepare(onnx_model) # runs the loaded model
ort_session = ort.InferenceSession('UltraLight-Face-Detector/ultra_light_640.onnx') # Loads model from database as a inferense 
# retrieves name of input layer as we need to use same name of input and output layers
input_name = ort_session.get_inputs()[0].name  

#five point face landmarking model
shape_predictor = dlib.shape_predictor('UltraLight-Face-Detector/shape_predictor_5_face_landmarks.dat')  
fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=256, desiredLeftEye=(0.35, 0.35))

video_capture = cv2.VideoCapture(0)

while True:
    cam, frame = video_capture.read()
    if frame is not None:
        h, w, _ = frame.shape

        # preprocess img acquired
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr to rgb
        img = cv2.resize(img, (640, 480)) # resize # this dimension is required for ultra_light_640.onnx
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        confidences, boxes = ort_session.run(None, {input_name: img})
        boxes, labels, probs = predict(w, h, confidences, boxes, 0.9)

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame=cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)

            #for locating eye and nose positions
            '''
            shape = shape_predictor(gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (80,18,236), -1)
            '''
            
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            #cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
            font = cv2.FONT_HERSHEY_PLAIN
            text = f"Face: {round(probs[i], 2)}"
            cv2.putText(frame, text, (x1, y2-3), font, 1, (255, 255, 255), 1)

        cv2.imshow('Window', frame)

    # Hit Enter on the keyboard to quit!
    if cv2.waitKey(1)==13:
        break

video_capture.release()
cv2.destroyAllWindows()