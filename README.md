# FTC Tensor Training scripts

## setup
```bash
virtualenv .
source bin/activate
pip install -r requirements.txt
```

## train a model
1. Create directory and place jpgs in it
```bash
mkdir training_raw
2. Annotate (label) images with PascalVOC format (xml files)
```bash
mkdir training_labels
labelImg
```
3. Split to training/testing folders
```bash
python shuffle_pascal_voc.py --source training_raw --annotation training_labels --target training
```
4. Create `training/labels.txt` file such that each line is like 'label:int'
5. Train model
```bash
python train_model.py -i training/train_img/ -a training/train_annotation/ -t training/test_img/ -s training/test_annotation -m model -l training/labels.txt
```
6. Test model
Using first camera:
```bash
python test_images.py -m model/model.tflite -c 0
````
Using image folder:
```bash
python test_images.py -m model/model.tflite -i training_raw
````
