
# Face detector and landmarks recognition

Copied from uncle Adrian <https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/> 

## Requirements
-   python 3.6+
-   pipenv

## Setup

En Ubuntu

    $ virtualenv -p /usr/bin/python3.7 env 
    $ source env/bin/activate
    $ pip install -r requirements.txt

Bajar los modelos entrenados necesarios ejecutando el script get_weights.sh

    $ sh get_weights.sh

## Execute
 	
	 $ python detect_face_parts.py -i images/smallfaces.png -p weights/shape_predictor_5_face_landmarks.dat
Hay dos parámetros por defecto modelo e imagen 

TODO:Tengo un bug en el cv2.imshow y a veces enseña imágenes vacías de 100px de ancho. No encuentro porque por eso guardo la imagen y veo que es problema del cv2.imshow()

## Face detection
Con dlib library que devuelve las coordenadas de todas las caras encontradas en la imagen

	detector = dlib.get_frontal_face_detector()

Se recomienda enviarle imágenes de un ancho de 500px y en escala de grises

## Landmarks recognition
hay dos modelos preentrenados disponibles: uno con 5 Landmarks (ojos) y otro con 64 puntos de la cara. Hay mas de 80 megas de diferencia entre ellos por lo que recomendamos usar el de 5 puntos weights/shape_predictor_5_face_landmarks.dat

### Descripción de los puntos devueltos

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

#For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])

### imutils import face_utils
Librería del tío Adrian para trabajar con imágenes de caras. Con diversas funciones helpers como face aligner. 
<https://github.com/jrosebr1/imutils/tree/master/imutils/face_utils>