import sys
import dlib
from skimage import io
import cv2
import os

# Take the image file name from the command line
file_name = sys.argv[1]
dir_name = '/openface/testing-images/for_test'

face_detector = dlib.get_frontal_face_detector()

win = dlib.image_window()

image = io.imread(file_name)

detected_faces = face_detector(image, 1)

print("Found {} faces in the file {}".format(len(detected_faces), file_name))


for i, face_rect in enumerate(detected_faces):

	
	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i+1, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
	
	x=face_rect.left()
	y=face_rect.right()
	h=face_rect.top()
	w=face_rect.bottom()
	
	temp=image[h-10:w+10, x-10:y+10]
	a,b,c = sys.argv[1].split(".")
	_,_,b = b.split("/")
	file_name = b+"face{}.jpg".format(i+1)
	fname = os.path.join(dir_name,file_name)
	cv2.imwrite(fname, temp)

