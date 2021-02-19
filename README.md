# 3D-RetinaNet for ROAD dataset
This repository contains code for 3D-RetinaNet, which is used a baseline for [ROAD dataset](https://github.com/gurkirt/road-dataset) in dataset release [paper](). It contains training and evaluation for ROAD and UCF-24 datasets. 


## Train 
Run the below

```
    python main.py --log_start=0 --log_step=1 --lr=0.005 --milestones=20000,25000 --max_iter=30000
```

