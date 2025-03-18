import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot access the camera")
else:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Cannot read frame")
    else:
        print("✅ Camera working correctly!")

cap.release()
