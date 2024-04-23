# Hackathon SuperAI SS4 Image Captioning

## Linux init

for linux users run init.sh for first time use
for windows users run these commands manually:

conda create -n "imageCaptioning" python=3.10.14
conda activate imageCaptioning
pip install --upgrade pip
pip install -r requirements.txt

mkdir / root/.kaggle/
mkdir dataset

then go download the dataset into /dataset

## TODO

1. Implement pipeline for fine-tuning YOLO on COCO images, and fine-tuning Text Translate in decoder part
2. Fine-tuning YOLO on [Thai food (classification task)](https://github.com/chakkritte/THFOOD-50), Thai [FoodyDudy (classification task)](https://www.kaggle.com/datasets/somboonthamgemmy/foodydudy), and [Food-500 Cap (Food image-captioning)](https://github.com/aaronma2020/Food500-Cap)
3. Classifier on what the dataset it is, to chose what model to use. (rounter)
