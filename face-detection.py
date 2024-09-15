import face_recognition
import cv2
#refrence persons
reference_images = [("Gaurav.jpg", "Gaurav Rajput"), ("Yash.jpg", "Ganesh"), ("Puneet.jpg", "SuperStarr")]

#encode all reference images
reference_encodings = []
reference_labels = []
for image_path, name in reference_images:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        reference_encodings.append(encoding[0])
        reference_labels.append(name)

# starting webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find face encodings in current frame
    face_encodings = face_recognition.face_encodings(rgb_frame)

    # Find  face locations for drawing rectangles
    face_locations = face_recognition.face_locations(rgb_frame)

    # for faces in current frame
    for (face_encoding, face_location) in zip(face_encodings, face_locations):
        # Compare faces with reference faces
        matches = face_recognition.compare_faces(reference_encodings, face_encoding)

        # Get location (top, right, bottom, left)
        top, right, bottom, left = face_location

        if any(matches):
            # Find index
            match_index = matches.index(True)
            name = reference_labels[match_index]
            # Draw a green box and label with the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # If the face doesn't match any reference face, draw a red box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "No Match", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time face Recognition', frame)

    # Breaking the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing
video_capture.release()
cv2.destroyAllWindows()
