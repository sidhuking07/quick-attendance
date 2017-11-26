import subprocess
import sys
import os

if len(sys.argv)>1:
	file = sys.argv[1]
img_file = str(file)

cmd1 = 'python find_faces.py '+img_file
cmd2 = './util/align-dlib.py ./testing-images/ align outerEyesAndNose ./testing-aligned-images/ --size 96'
cmd3 = './batch-represent/main.lua -outDir ./testing-generated-embeddings/ -data ./testing-aligned-images/'
cmd4 = 'python classifier_predicts.py'

cmd = []
cmd.append(cmd1)
cmd.append(cmd2)
cmd.append(cmd3)
cmd.append(cmd4)

for i in range(len(cmd)):
	os.system(cmd[i])
