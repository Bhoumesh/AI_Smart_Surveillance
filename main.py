import cv2
import datetime
import os

# ===============================
# Setup
# ===============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

os.makedirs("captures", exist_ok=True)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

# ===============================
# Main Loop
# ===============================
while cap.isOpened():

    # -------- Motion Detection --------
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        motion_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # -------- Face Detection --------
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame1, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)

    # -------- Timestamp --------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame1, timestamp,
        (10, frame1.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2
    )

    # -------- Save Image --------
    if motion_detected and len(faces) > 0:
        filename = f"captures/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame1)

    cv2.imshow("AI Smart Surveillance System", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
