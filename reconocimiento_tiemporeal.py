import tensorflow as tf
import numpy as np
import cv2

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = tf.keras.models.load_model('NN_for_my_face3.h5')

def get_className(classNo):
    if classNo == 1:
        return 'Alan'


while True:
	sucess, imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
	for x, y, w, h in faces:
		crop_img = imgOrignal[y:y+h,x:x+h]
		img = cv2.resize(crop_img, (192, 192))
		img = img.reshape(1, 192, 192, 3)
		prediction = model.predict(img)
		classIndex = np.where(prediction > .85, 1, 0)
		probabilityValue = np.amax(prediction)
		if classIndex == 0:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0, 0, 255),2)
			cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (0, 0, 255), -2)
			cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
		elif classIndex == 1:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40), (x+w, y), (0, 255, 0),-2)
			cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

		cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
	cv2.imshow("Result", imgOrignal)
	k = cv2.waitKey(1)
	if k == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()