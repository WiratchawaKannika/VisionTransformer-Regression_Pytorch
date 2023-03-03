# VisionTransformer Regression : Pytorch

- ### Conda environment
```
conda activate Vit
```

# Training model on Rheology 2023 Dataset

- [x] Customizing Models :point_right: model.py 

```
print("Available Vision Transformer Models: ")
timm.list_models("vit*")
```

- [x] Data constants :point_right: datasets.py 

```
## data constants
BATCH_SIZE = 64
NUM_WORKERS = 0
IMG_SIZE = 384
```

- [x] Training model 

```
python train.py --epochs 2 --gpu 1 --base_path /home/kannika/codes_AI/Rheology2023 --save_dir /media/SSD/rheology2023/VitModel/Regression --name ExpTest
```


- [x] Resume model 

```
python train.py --epochs 2 --gpu 1 --base_path /home/kannika/codes_AI/Rheology2023 --save_dir /media/SSD/rheology2023/VitModel/Regression --name ExpTestResume --resume --modelPATH /media/SSD/rheology2023/VitModel/Regression/ExpTest/weight/RheoVitRegress_Entire_R1model.pth
```


- [x] TensorBoard PyTorch
```
tensorboard --logdir /path/to/tensorboard_logs/ --bind_all
```

- [The SummaryWriter class is your main entry to log data for consumption and visualization by TensorBoard](https://pytorch.org/docs/stable/tensorboard.html#)
