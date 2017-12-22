import sys
import dlib
from skimage import io

predictor_model = "shape_predictor_68_face_landmarks.dat"

file_name = sys.argv[1] #take image from command

face_detector = dlib.get_frontal_face_detector() 
face_pose_predictor = dlib.shape_predictor(predictor_model)

win = dlib.image_window()

file_name = sys.argv[1] 

image = io.imread(file_name) 

detected_faces = face_detector(image, 1) #run HOG face detector

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

win.set_image(image) 


for i, face_rect in enumerate(detected_faces):
	
	win.add_overlay(face_rect) #print a box around every detected face

	pose_landmarks = face_pose_predictor(image, face_rect) 

	win.add_overlay(pose_landmarks) 
	        
dlib.hit_enter_to_continue()
