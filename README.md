# nunbody_segmentation
conda create -n nunbody python=3.8.8 -y 
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y 
pip install matplotlib tqdm pycocotools opencv-python pytorch_lightning albumentations segmentation-models-pytorch wandb

# crf
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
pip install tqdm pandas