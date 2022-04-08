from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import face_recognition
import cv2
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import messaging

token = [
    "cVM3iyH7QmK2zQxJczHomf:APA91bGZ5eOMkYeGyoIw36RsDG6L__Ouv68VakxWm7KiF0FVUWKUyhX3ziA-0o0D7Bulc_DWeZ_rQnMwnDZCn6MxaR9ahnqgjOEnNBfRYqcOPKG4dwsYTgLcF_hx9mHqpw0d7bCuUdDk",
    "f28_DI-STnmMiGHzWDd6P8:APA91bEMT-2jOtdVajWHtu3WgKlTD7WMB-43I80W85qrDoeRmxuFz-N0_gEpgPVAtdfXi8RfKMRHgJGPLDzbsDlXqMMWK3JtcgOclFdc3FUUznis2mkUZKUA-ilsLODY6Du6GXcrMiHb"]

cred = credentials.Certificate("C:/Users/harsh/Downloads/mask-check-d53b8-firebase-adminsdk-rhycc-851b989cb1.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
users_ref = db.collection('users')


def sendPush(title, msg, img, registration_token, dataObject):
    # See documentation on defining a message payload.
    message = messaging.MulticastMessage(
        notification=messaging.Notification(
            title=title,
            body=msg,
            image=img
        ),
        data=dataObject,
        tokens=registration_token,
    )
    response = messaging.send_multicast(message)



def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations, and the list of predictions from our face mask
    # network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.model")

image1 = face_recognition.load_image_file(
    os.path.abspath("C:/Users/harsh/PycharmProjects/Face-Mask-Detection/recognize/images/saloni.jpeg"))
image1_face_encoding = face_recognition.face_encodings(image1)[0]

image2 = face_recognition.load_image_file(
    os.path.abspath("C:/Users/harsh/PycharmProjects/Face-Mask-Detection/recognize/images/harsh.jpg"))
image2_face_encoding = face_recognition.face_encodings(image2)[0]

image3 = face_recognition.load_image_file(
    os.path.abspath("C:/Users/harsh/PycharmProjects/Face-Mask-Detection/recognize/images/praju.jpeg"))
image3_face_encoding = face_recognition.face_encodings(image3)[0]

image4 = face_recognition.load_image_file(
    os.path.abspath("C:/Users/harsh/PycharmProjects/Face-Mask-Detection/recognize/images/gitty.jpeg"))
image4_face_encoding = face_recognition.face_encodings(image4)[0]

image5 = face_recognition.load_image_file(
    os.path.abspath("C:/Users/harsh/PycharmProjects/Face-Mask-Detection/recognize/images/kiransir.jpg"))
image5_face_encoding = face_recognition.face_encodings(image5)[0]

image6 = face_recognition.load_image_file(
    os.path.abspath("C:/Users/harsh/PycharmProjects/Face-Mask-Detection/recognize/images/kaustubh.jpeg"))
image6_face_encoding = face_recognition.face_encodings(image6)[0]

image7 = face_recognition.load_image_file(
    os.path.abspath("C:/Users/harsh/PycharmProjects/Face-Mask-Detection/recognize/images/vishalsir.png"))
image7_face_encoding = face_recognition.face_encodings(image7)[0]


known_face_encodings = [
    image1_face_encoding,
    image2_face_encoding,
    image3_face_encoding,
    image4_face_encoding,
    image5_face_encoding,
    image6_face_encoding,
    image7_face_encoding
]
known_face_names = [
    "Saloni Rane",
    "Harsh Saraiya",
    "Prajakta Mhaske",
    "Saurav Hiwanj",
    "Kiran Deshpande",
    "Kaustubh Sawant",
    "Prof Vishal Badgujar"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
total_faces = []
total_faces_list = []
dictkey = ('Name')

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # detect faces in the frame and determine if they are wearing a face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        if label == "No Mask":
            if process_this_frame:
                face_locations = face_recognition.face_locations(frame)
                # print(face_locations)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                # print(face_encodings)
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    face_names.append(name)
                    print("Face detected -- {}".format(face_names))
                    total_faces.extend(face_names)
                    total_faces_list = list(dict.fromkeys(total_faces))
                    # print(total_faces_list)

        process_this_frame = not process_this_frame

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # print(j.to_dict())

    for i in total_faces_list:
        result = users_ref.where("name", "==", i).get()
        for j in result:
            output = j.to_dict()
            # print(output)
            sendPush("ALERT", "{} was caught without a mask!".format(output.get("name")),
                     output.get("profilepicurl"), token, output)
            break
        break

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# print(total_faces_list)
# for i in total_faces_list:
#     result = users_ref.where("name", "==", i).get()
#     for j in result:
#         output = j.to_dict()
#         # print(output)
#         sendPush("ALERT", "{} was caught without a mask!".format(output.get("name")),
#                  output.get("profilepicurl"), token, output)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
