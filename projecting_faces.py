import sys
import dlib
import cv2
import openface

predictor_model = "shape_predictor_68_face_landmarks.dat"

file_name = sys.argv[1] #take input from command

face_detector = dlib.get_frontal_face_detector() #create HOG face detector from dlib
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model) #use AlignDlib from openface for alignment

file_name = sys.argv[1] #take the input from command

image = cv2.imread(file_name) #load image as array

detected_faces = face_detector(image, 1) #run HOG face detector


# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

	pose_landmarks = face_pose_predictor(image, face_rect) #get the pose

	alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)  #perfrom alignment

	cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace) #save the aligned image
