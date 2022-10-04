#!/usr/bin/env python3

# split data into training and testing set
import os, random, shutil, argparse, pathlib

parser = argparse.ArgumentParser(description='Shuffle images for tenslorflow training')
parser.add_argument('-s', '--source', type=pathlib.Path, dest='source_img_path', help='Path to shuffle images from', required=True)
parser.add_argument('-a', '--annoation', type=pathlib.Path, dest='source_voc_path', help='Path to shuffle images annotations from', required=True)
parser.add_argument('-t', '--target', type=pathlib.Path, dest='train_path', help='Path to create training images and annotations', required=True)
parser.add_argument('-r', '--ratio', type=float, dest='train_ratio', help='Train ratio', default=0.8, required=False)
parser.add_argument('-f', '--format', dest='extension', help='file type', default='jpg')
args = parser.parse_args()

def create_if_needed_test_if_empty(path:pathlib.Path):
	if not path.exists():
		path.mkdir(parents=True)
	elif not path.is_dir():
		raise Exception(f"Path {path} should be a directory")

create_if_needed_test_if_empty(args.train_path)

image_paths = os.listdir(args.source_img_path.resolve())
random.shuffle(image_paths)

source_img_path = args.source_img_path.resolve()
source_voc_path = args.source_voc_path.resolve()
train_img_path = args.train_path.joinpath("train_img")
train_voc_path = args.train_path.joinpath("train_annotation")
test_img_path = args.train_path.joinpath("test_img")
test_voc_path = args.train_path.joinpath("test_annotation")
create_if_needed_test_if_empty(train_img_path)
create_if_needed_test_if_empty(train_voc_path)
create_if_needed_test_if_empty(test_img_path)
create_if_needed_test_if_empty(test_voc_path)

file_train_count = int(len(image_paths) * args.train_ratio)
print(f"training set size is {file_train_count}")
for i, image_path in enumerate(image_paths):
	img_path = f'{source_img_path}/{image_path}'
	voc_path = f'{source_voc_path}/{image_path.replace(args.extension, "xml")}'
	if not os.path.exists(voc_path):
		print(f"skipping {voc_path}")
	else:
		if i < file_train_count:
			shutil.copy(img_path, train_img_path)
			shutil.copy(voc_path, train_voc_path)
		else:
			shutil.copy(img_path, test_img_path)
			shutil.copy(voc_path, test_voc_path)
