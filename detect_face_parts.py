#LANDMARKS definition are in face_utils from imutils
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, 
	default='weights/shape_predictor_68_face_landmarks.dat',
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=False,
	default='images/smallfaces.png',
	help="path to input image")
args = vars(ap.parse_args())


def show_landmarks(image,rects):
	#Show all landamarks in all faces
	for (i, face) in enumerate(rects):
			x, y, w, h = face.left(), face.top(), face.width(), face.height()			
			# Make the prediction and transfom it to numpy array
			shape = predictor(gray, face)
			shape = face_utils.shape_to_np(shape)
			# Draw on our image, all the finded cordinate points (x,y)
			for (x, y) in shape:
				cv2.circle(image, (x, y), 2, (230, 255, 0), -1)
	return image


def show_centroid(image,rects):
	#Show all landamarks in all faces
	for (i, face) in enumerate(rects):
			x, y, w, h = face.left(), face.top(), face.width(), face.height()
			#For each face detect landmarks
			shape = predictor(gray, face)
			shape = face_utils.shape_to_np(shape)

			if (len(shape)==68):
				# extract the left and right eye (x, y)-coordinates
				(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
				(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
			else:
				(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_5_IDXS["left_eye"]
				(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_5_IDXS["right_eye"]

			leftEyePts = shape[lStart:lEnd]
			rightEyePts = shape[rStart:rEnd]
			# compute the center of mass for each eye
			leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
			rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
			# compute center (x, y)-coordinates (i.e., the median point)
			# between the two eyes in the input image			
			eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
				(leftEyeCenter[1] + rightEyeCenter[1]) // 2)
			print('leftEyePts ,rightEyePts, eyesCenter',leftEyeCenter ,rightEyeCenter,eyesCenter)			

			# Draw on our image, all the finded cordinate points (x,y)
			#for (x, y) in leftEyeCenter:
			x,y=leftEyeCenter
			cv2.circle(image, (x,y), 2, (230, 255, 0), -1)
			x,y=rightEyeCenter
			cv2.circle(image, (x,y), 2, (0, 255, 230), -1)
			x,y=eyesCenter
			cv2.circle(image, (x,y), 2, (0, 255, 0), -1)
	return image



if __name__ == "__main__":    
	# initialize dlib's face detector (HOG-based)
	detector = dlib.get_frontal_face_detector()
	# initialitze the facial landmark predictor
	predictor = dlib.shape_predictor(args["shape_predictor"])
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(args["image"])	
	print('height, width', image.shape[:2])
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	print('Coordenadas Caras:',rects)
	image = show_centroid(image,rects)
	cv2.imshow("Entrecejo", image)
	cv2.waitKey(0)
	cv2.imwrite('entrecejo.png',image)
	image = show_landmarks(image,rects)
	cv2.imshow("Todos los puntos", show_landmarks(image,rects))
	cv2.waitKey(0)
	cv2.imwrite('todos_los_Landmarks.png',image)
	cv2.destroyAllWindows()
	
	