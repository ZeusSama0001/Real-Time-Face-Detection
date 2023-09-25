import cv2
import time

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize variables for frame rate calculation
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray)

        for face in faces:
            [x, y, w, h] = face
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0))

        cv2.imshow("My Frame", frame)

        # Increment frame count for frame rate calculation
        frame_count += 1

    key = cv2.waitKey(10)

    if key == ord("q"):
        break

# Calculate frame rate and display it
end_time = time.time()
elapsed_time = end_time - start_time
frame_rate = frame_count / elapsed_time
print(f"Frame rate: {frame_rate:.2f} fps")

cap.release()
cv2.destroyAllWindows()
