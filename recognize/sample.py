import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import messaging

cred = credentials.Certificate("C:/Users/harsh/Downloads/mask-check-d53b8-firebase-adminsdk-rhycc-851b989cb1.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

total_faces = ['Pranav Thanekar', 'Saurav Hiwanj']
token = [
    "cVM3iyH7QmK2zQxJczHomf:APA91bGZ5eOMkYeGyoIw36RsDG6L__Ouv68VakxWm7KiF0FVUWKUyhX3ziA-0o0D7Bulc_DWeZ_rQnMwnDZCn6MxaR9ahnqgjOEnNBfRYqcOPKG4dwsYTgLcF_hx9mHqpw0d7bCuUdDk",
    "f28_DI-STnmMiGHzWDd6P8:APA91bEMT-2jOtdVajWHtu3WgKlTD7WMB-43I80W85qrDoeRmxuFz-N0_gEpgPVAtdfXi8RfKMRHgJGPLDzbsDlXqMMWK3JtcgOclFdc3FUUznis2mkUZKUA-ilsLODY6Du6GXcrMiHb"]


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

    # Send a message to the device corresponding to the provided
    # registration token.
    response = messaging.send_multicast(message)
    # Response is a message ID string.
    print('Successfully sent message:', response)


users_ref = db.collection('users')
for i in total_faces:
    result = users_ref.where("name", "==", i).get()
    for j in result:
        output = j.to_dict()
        print(output)
        sendPush("ALERT", "{} was caught without a mask!".format(output.get("name")), output.get("profilepicurl"), token, output)
