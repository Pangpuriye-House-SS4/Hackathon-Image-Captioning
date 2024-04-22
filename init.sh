#prepping environment
conda create -n "imageCaptioning" python=3.10.14
conda activate imageCaptioning
pip install --upgrade pip
pip install -r requirements.txt

#prepping directories
mkdir / root/.kaggle/
mkdir dataset
cd dataset

#downloading files
pip install gdown
gdown https://drive.google.com/file/d/1wz5s4FJYrT3CWDeDhS9WfzZD64cQtFf9
unrar e /content/drive/MyDrive/Dataset/Datasets.rar