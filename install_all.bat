@echo off

echo Installing requirements.txt...
pip install -r requirements.txt

echo Downloading model...
gdown --id 1Zgz4wH6MGXwg5Y4Z-a8mPBrm5rcUvYIf --output best.pth

echo Installing PyTorch with CUDA 11.7 support...
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

echo Installation complete!
pause
