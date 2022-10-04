# FTC Tensor Training scripts

Here are the steps to take a video or a set of stills and create a tflite file to use with OnBot Java or Blocks for an FTC game.

## setup
```bash
virtualenv .
source bin/activate
pip install -r requirements.txt
```

## train a model
1. Create directory and place jpgs in (must be .jpg)
```bash
mkdir training_raw
```
You can grab stills from a video using something like this command:
`ffmpeg -i bolt_small.mp4 training_raw/bolt_small_%05d.jpg`

2. Annotate (label) images with PascalVOC format (xml files)
```bash
mkdir annotation_dir
labelImg
```
After starting label image, go to File, pick 'Change Save Dir' and pick where you want to save the annotations for the images (`annotation_dir` from the previous step).
Next, pick 'Open Dir' from the File menu
For each image, press 'W' to start labeling.
'A' and 'D' move forward and backwards in the images.
You can change images while the label select tool is active.
Stay tight around the objects.
Once through, hold A and W to scroll through the images and make sure you have consistent boxing.

3. Split to training/testing folders
This will randomly pick images to use for 
```bash
python shuffle_pascal_voc.py --source training_raw_dir --annotation annotation_dir --target training_dir
```
4. Create `training/labels.txt` file such that each line is like ```
```
label:int
```
where `label` is the same name from the image labeling step #2, you can confirm by opening some of the XML files in annotation_dir.  The `int` is a natural number, probably starting at 1.

5. Train model
```bash
python train_model.py -i training/train_img/ -a training/train_annotation/ -t training/test_img/ -s training/test_annotation -m model -l training/labels.txt
```
`model` will be a directory with `labels.txt`, `model.tflite`, and a directory called `saved_model` with the training data to allow for retraining.

6. Test model
Using first camera:
```bash
python test_images.py -m model/model.tflite -c 0
````
Using image folder:
```bash
python test_images.py -m model/model.tflite -i training_raw
````

Now you should be able to take this file and upload to the robot and use it.