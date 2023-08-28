 import cv2
import numpy as np


net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
         class_name = class_name.strip()
         classes.append(class_name)

print("objects list")
print(classes)


#  initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


button_person = False

def click_button(event, x, y, flags, params):
#     global  button_person
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, y)
#         polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
#
#         is_inside = cv2.pointPolygonTest(polygon, (x, y), False)
#         if is_inside > 0:
#             print("we'r clicking inside the button", x, y)
#
#             if button_person is False:
#                 button_person = True
#             else:
#                 button_person = False
#
#             print("now button is: ", button_person)
#
#
# #create window
# cv2.namedWindow("Frame")
# cv2.setMouseCallback("Frame", click_button)
#
# while True:
#     # Get frames
#     ret, frame = cap.read()
#
#     # object detection
#     (class_ids, scores, bboxes) = model.detect(frame)
#     for class_id, score, bbox in zip(class_ids, scores, bboxes):
#         (x, y, w, h ) = bbox
#         class_name = classes[class_id]
#         # print(x, y, w, h)
#         # print(class_name)
#         if class_name == "person" and button_person is True:
#             cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)
#
#     # print("class ids", class_ids)
#     # print("scores", scores)
#     # print("bboxes",bboxes)
#
#     #create button
#     # cv2.rectangle(frame, (20, 20), (215, 70), (0, 0, 200), -1)
#     polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
#     cv2.fillPoly(frame, polygon, (0, 0, 200))
#     cv2.putText(frame, "person", (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
#
#
#
#     cv2.imshow("Frame", frame)
#     cv2.waitKey(1)