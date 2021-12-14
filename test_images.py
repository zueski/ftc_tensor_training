#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import cv2
import pathlib
import argparse

parser = argparse.ArgumentParser(description='Show detections for folder')
parser.add_argument('-i', '--images', type=pathlib.Path, dest='images_path', help='Path to shuffle images from', required=False, default="training_img")
parser.add_argument('-m', '--model', type=str, dest='model_path', help='Path to shuffle images annotations from', required=True, default="models/model.tflite")
parser.add_argument('-l', '--labels', type=pathlib.Path, dest='labels_path', help='Path to shuffle images annotations from', required=False)
parser.add_argument('-c', '--use-camera', type=int, dest='use_camera', help='Index for camera, set to 0 if unknown', required=False, default=-1)
parser.add_argument('--camera-path', type=int, dest='camera_path', help='Path to video device (linux)', required=False)
parser.add_argument('-v', '--verbose', dest='debug', help='Verbose debug', action='store_true')
args = parser.parse_args()


def draw_rect(image, box, label, score):
	color = (255, 255, 255)
	y_min = int(max(1, (box[0] * image.shape[0])))
	x_min = int(max(1, (box[1] * image.shape[1])))
	y_max = int(min(image.shape[0], (box[2] * image.shape[0])))
	x_max = int(min(image.shape[1], (box[3] * image.shape[1])))
	# draw a rectangle on the image
	cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
	cv2.putText(image, f'{int(label)}: {score}', (x_min, y_max), 1, 2, color, 1)

def display_image(img, interpreter, output_details):
	new_img = cv2.resize(img, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
	interpreter.set_tensor(input_details[0]['index'], [new_img])

	interpreter.invoke()
	i = 0
	if args.debug:
		for output in output_details:
			print(f"output {i}: {interpreter.get_tensor(output['index'])}")
			i=i+1
	rects = interpreter.get_tensor(output_details[1]['index'])
	labels = interpreter.get_tensor(output_details[3]['index'])
	scores = interpreter.get_tensor(output_details[0]['index'])
	
	for index, score in enumerate(scores[0]):
		if score > 0.2:
			draw_rect(img, rects[0][index], labels[0][index], scores[0][index])
	cv2.imshow("image", img)




if args.debug:
	print("opening model", args.model_path)
interpreter = tf.lite.Interpreter(model_path=args.model_path)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if args.debug:
	print('Found signature:', interpreter.get_signature_list())
	print('Found input:', input_details)
	print('Found output:', output_details)

interpreter.allocate_tensors()
print("Press 'q' to end")
if args.use_camera > -1:
	if args.camera_path is not None:
		vid = cv2.VideoCapture(args.camera_path)
	else:
		vid = cv2.VideoCapture(args.use_camera)
	while(True):
		ret, frame = vid.read()
		if ret:
			display_image(frame, interpreter, output_details)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	vid.release()
else:
	for file in args.images_path.iterdir():
		if file.suffix != '.jpg' and file.suffix != '.png':
			if args.debug:
				print("Skipping file ", file)
			continue
		if args.debug:
			print("reading file", file.resolve())
		img = cv2.imread(r"{}".format(file.resolve()))
		display_image(img, interpreter, output_details)
		if cv2.waitKey(0) & 0xFF == ord('q'):
			break
cv2.destroyAllWindows()