#!/usr/bin/env python3

import numpy as np
import os
import re
import pathlib
import argparse

# https://www.tensorflow.org/lite/tutorials/model_maker_object_detection


parser = argparse.ArgumentParser(description='Trains model from PascalVOC/xml annotation')
parser.add_argument('-i', '--images-train', type=pathlib.Path, dest='source_img_path', help='Path for training images', required=True)
parser.add_argument('-a', '--annotation-train', type=pathlib.Path, dest='source_voc_path', help='Path for training xml files', required=True)
parser.add_argument('-t', '--images-test', type=pathlib.Path, dest='test_img_path', help='Path for validation/testing images', required=True)
parser.add_argument('-s', '--annotation-test', type=pathlib.Path, dest='test_voc_path', help='Path for validation/testing annotations', required=True)
parser.add_argument('-l', '--label', type=pathlib.Path, dest='label_file', help='test file with labels', required=True)
parser.add_argument('-m', '--model', type=pathlib.Path, dest='model_output', help='Path to write model files to', required=True)
parser.add_argument('-p', '--spec', type=str, dest='model_spec', help='spec type', required=False, default='efficientdet_lite0', choices=['efficientdet_lite', 'efficientdet_lite0','efficientdet_lite1','efficientdet_lite2','efficientdet_lite3','efficientdet_lite4'])
args = parser.parse_args()

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get(args.model_spec)

label_map = dict()
with open(args.label_file.resolve()) as file:
	index_pattern = re.compile("([^:]+):([0-9]+)")
	for line in file:
		l = index_pattern.match(line)
		if l:
			label_map[int(l[2])] = l[1]
print("found labels:", label_map)

train_data = object_detector.DataLoader.from_pascal_voc(
	images_dir=args.source_img_path.resolve().as_posix(),
	annotations_dir=args.source_voc_path.resolve().as_posix(),
	label_map=label_map)

validation_data = object_detector.DataLoader.from_pascal_voc(
	images_dir=args.test_img_path.resolve().as_posix(),
	annotations_dir=args.test_voc_path.resolve().as_posix(),
	label_map=label_map)

model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)

print(f"model is { type(model) }")


print(f"evaluate: {model.evaluate(validation_data)}")

#config = QuantizationConfig.for_dynamic(representative_data=train_data)

# Export to Tensorflow Lite model and label file in `export_dir`.
model.export(export_dir=args.model_output.resolve(),export_format=ExportFormat.TFLITE)
model.export(export_dir=args.model_output.resolve(),export_format=ExportFormat.LABEL)
model.export(export_dir=args.model_output.resolve(),export_format=ExportFormat.SAVED_MODEL)
print(model.summary())
