import cv2
import os
import imutils

#Definimos el nombre de la carpeta en la cual se alamcenarán los rostros
personName = 'data/fotos_Alan/'
#Definimos la ruta en la cual se guardará la carpeta con las fotos
dataPath = 'data/'
personPath = dataPath + personName

#Creamos el directorio con el nombre de la perosona 
if not os.path.exists(personPath):
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath)

#Indicamos que tomaremos un vídeo en directo
#cap = cv.VideoCapture(0, cv.Cap_DSHOW)
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('Video.mp4')

#Utilizamos el clasificador de rostros frontales de haarcasdade 
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
count = 0

#Iniciamos un ciclo while para comparar cada uno de los fotogramas del vídeo
while True:  
    ret, frame = cap.read()
    if ret == False: break

#Redimensionamos el tamaño de los fotogramas del vídeo de entrada con imutils
    frame =  imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

#Se obtinene las coordenadas de los rostros mediante el clasificador
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

#Se definen las coordenadas de los rectángulos a partir de los datos adquiridos en faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro, (192, 192), interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        count = count + 1
    cv2.imshow('frame', frame)

#INdicamos que el codigo se rompa una vez que el contador llegue a 300, es decir una vez
#obtenidas 300 capturas
    k = cv2.waitKey(1)
    if k == 27 or count >= 1000:
        break

cap.release()
cv2.destroyAllWindows()