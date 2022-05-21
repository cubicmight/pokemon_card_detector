import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
