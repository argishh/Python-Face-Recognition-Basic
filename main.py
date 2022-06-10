import numpy as np
import cv2
import pickle
print("Program Running...")
faceCascade = cv2.CascadeClassifier('.\cascade\haarcascade_frontalface.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./trained/face-trainner.yml")

labels = {"John_Mayer": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x, y, w, h) in faces:
		roi_gray = gray[y:y+h, x:x+w]
		id_, conf = recognizer.predict(roi_gray)
		if conf>=4 and conf <= 85:
			font, name = cv2.FONT_HERSHEY_COMPLEX, labels[id_]
			color, stroke = (255, 255, 255), 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		color, stroke = (255, 0, 0), 2
		end_cord_x, end_cord_y  = x + w, y + h
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
	cv2.imshow('Video Feed',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
print("Program Ended...")