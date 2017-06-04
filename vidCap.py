import cv2

cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()
	cv2.imshow('original', frame)
	k = cv2.waitKey(1) & 0xFF
	if k==ord('q'):
		break
	if k==ord('c'):
		cv2.imwrite('./test.jpg', frame)
cap.release()
cv2.destroyAllWindows()
