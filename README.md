# 3D-RetinaNet for ROAD and UCF-24 dataset
This repository contains code for 3D-RetinaNet, a novel Single-Stage action detection newtwork proposed along with [ROAD dataset](https://github.com/gurkirt/road-dataset).  Our [TPAMI paper](https://www.computer.org/csdl/journal/tp/5555/01/09712346/1AZL0P4dL1e) contain detailed description 3D-RetinaNet and ROAD dataset. This code contains training and evaluation for ROAD and UCF-24 datasets. 



## Table of Contents
- <a href='#requirements'>Requirements</a>
- <a href='#training-3d-retinanet'>Training 3D-RetinaNet</a>
- <a href='#testing-and-building-tubes'>Testing and Building Tubes</a>
- <a href='#performance'>Performance</a>
- <a href='#todo'>TODO</a>
- <a href='#citation'>Citation</a>
- <a href='#references'>Reference</a>


## Requirements
We need three things to get started with training: datasets, kinetics pre-trained weight, and pytorch with torchvision and tensoboardX. 

### Dataset download an pre-process

- We currently only support following two dataset.
    - [ROAD dataset](https://github.com/gurkirt/road-dataset) in dataset release [paper](https://arxiv.org/pdf/2102.11585.pdf)
    - [UCF24](http://www.thumos.info/download.html) with [revised annotations](https://github.com/gurkirt/corrected-UCF101-Annots) released with our [ICCV-2017 paper](https://arxiv.org/pdf/1611.08563.pdf).

- Visit [ROAD dataset](https://github.com/gurkirt/road-dataset) for download and pre-processing. 
- You can download `rgb-images` it from my [google drive link](https://drive.google.com/file/d/1o2l6nYhd-0DDXGP-IPReBP4y1ffVmGSE/view?usp=sharing) for UCF24 Dataset. Download annotations from [corrected-UCF10-annots-repo](https://github.com/gurkirt/corrected-UCF101-Annots/blob/master/pyannot_with_class_names.pkl). 
    - Your data directory should look like:
        ```
        - ucf24/
            - pyannot_with_class_names.pkl
            - rgb-images
                - class-name ...
                    - video-name ...
                        - images ......
        ```
### Pytorch and weights

  - Install [Pytorch](https://pytorch.org/) and [torchvision](http://pytorch.org/docs/torchvision/datasets.html)
  - INstall tensorboardX viad `pip install tensorboardx`
  - Pre-trained weight on [kinetics-400](https://deepmind.com/research/open-source/kinetics). Download them by changing current directory to `kinetics-pt` and run the bash file [get_kinetics_weights.sh](./kinetics-pt/get_kinetics_weights.sh). OR Download them from  [Google-Drive](https://drive.google.com/drive/folders/1xERCC1wa1pgcDtrZxPgDKteIQLkLByPS?usp=sharing). Name the folder `kinetics-pt`, it is important to name it right. 



## Training 3D-RetinaNet
- We assume that you have downloaded and put dataset and pre-trained weight in correct places.    
- To train 3D-RetinaNet using the training script simply specify the parameters listed in `main.py` as a flag or manually change them.

You will need 4 GPUs (each with at least 10GB VRAM) to run training.

Let's assume that you extracted dataset in `/home/user/road/` and weights in `/home/user/kinetics-pt/` directory then your train command from the root directory of this repo is going to be:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /home/user/ /home/user/  /home/user/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041
```

Second instance of `/home/user/ ` in above command specifies where checkpoint weight and logs are going to be stored. In this case, checkpoints and logs will be in `/home/user/road/cache/<experiment-name>/`.

Different parameters in `main.py` will result in different performance. Validation split is automatically selected based in training split number in road.

You can train `ucf24` dataset by change some command line parameter as the training sechdule and learning rate differ compared ot `road` training.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /home/user/ /home/user/  /home/user/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=ucf24 --TRAIN_SUBSETS=train --VAL_SUBSETS=val --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10
```

- Training notes:
  * Network occupies almost 9.7GB VRAM on each GPU, we used 1080Ti for training and normal training takes about 24 hrs on road dataset.
  * During training checkpoint is saved every epoch also log it's frame-level `frame-mean-ap` on a subset of validation split test.
  * Crucial parameters are `LR`, `MILESTONES`, `MAX_EPOCHS`, and `BATCH_SIZE` for training process.
  * `label_types` is very important variable, it defines label-types are being used for training and validation time it is bummed up by one with `ego-action` label type. It is created in `data\dataset.py` for each dataset separately and copied to `args` in `main.py`, further used at the time of evaluations.
  * Event detection and triplet detection is used interchangeably in this code base. 

## Testing and Building Tubes
To generate the tubes and evaluate them, first, you will need frame-level detection and link them. It is pretty simple in out case. Similar to training command, you can run following commands. These can run on single GPUs. 

There are various `MODEs` in `main.py`. You can do each step independently or together. At the moment `gen-dets` mode generates and evaluated frame-wise detection and finally performs tube building and evaluation.

For ROAD dataset, run the following commands.

```
python main.py /home/user/ /home/user/  /home/user/kinetics-pt/ --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 
```

and for UCF24


```
python main.py /home/user/ /home/user/  /home/user/kinetics-pt/ --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=ucf24 --TRAIN_SUBSETS=train --VAL_SUBSETS=val --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.00245 --EVAL_EPOCHS=10 --GEN_NMS=80 --TOPK=20 --PATHS_IOUTH=0.25 --TRIM_METHOD=indiv
```

- Testing notes
  * Evaluation can be done on single GPU for test sequence length up to 32  
  * No temporal trimming is performed for ROAD dataset however we use class specific alphas with temporal trimming formulation described in [paper](https://arxiv.org/pdf/2102.11585.pdf), which relies on temporal label consistency. 
  * Please go through the hypermeter in `main.py` to understand there functions.
  * After performing tubes a detection `.json` file is dumped, which is used for evaluation, see `tubes.py` for more detatils.
  * See `modules\evaluation.py` and `data\dataset.py` for frame-level and video-level evaluation code to compute `frame-mAP` and `video-mAP`.


## Performance

Here, you find the reproduced  results from our [paper](https://arxiv.org/pdf/2102.11585.pdf). We use training split #3 for reproduction on a different machines compared to where results were generated for the paper. Below you will find the test results on validation split #3, which closer to test set compared to other split in terms of environmental conditions.
We there is little change in learning rate here, so results are little different than the paper. Also, there are six tasks in ROAD dataset that makes it difficult balance the learning among tasks.

Model is set to `I3D` with `resnet50` backbone. Kinetics pre-trained weights used for `resnet50I3D`, download link to given above in <a href=#requirements> Requirements section</a>. Results on split #3 with test-sequence length being 8 `<frame-AP@0.5>/<video-mAP@0.2>`. 



<table style="width:100% th">
  <tr>
    <td>Model</td>
    <td>I3D</td> 
    <!-- <td>I3D</td>
    <td>0.75</td>
    <td>0.5:0.95</td>
    <td>frame-mAP@0.5</td>
    <td>accuracy(%)</td> -->
  </tr>
  <tr>
    <td align="left">Agentness</td> 
    <td>54.7/--</td>
    <!-- <td>32.07</td>
    <td>00.85</td> 
    <td>07.26</td>
    <td> -- </td> 
    <td> -- </td> -->
  </tr>
  <tr>
    <td align="left">Agent</td> 
    <td>31.1/26.0</td>
    <!-- <td>32.07</td>
    <td>00.85</td> 
    <td>07.26</td>
    <td> -- </td> 
    <td> -- </td> -->
  </tr>
  <tr>
    <td align="left">Action</td> 
    <td>22.0/16.1</td>
    <!-- <td>36.37</td> 
    <td>07.94</td>
    <td>14.37</td>
    <td> -- </td>
    <td> -- </td> -->
  </tr>
  <tr>
    <td align="left">Location</td> 
    <td>27.3/24.2</td>
    <!-- <td>43.00</td> 
    <td>14.10</td>
    <td>19.20</td>
    <td> -- </td>
    <td> -- </td> -->
  </tr>
  <tr>
    <td align="left">Duplexes </td> 
    <td>23.7/19.5</td>
    <!-- <td>46.30</td>
    <td>15.00</td> 
    <td>20.40</td>
    <td> -- </td>
    <td> 91.12 </td>   -->
  </tr>
  <tr>
    <td align="left">Events/triplets </td> 
    <td>13.9/15.5</td>
    <!-- <td>40.59</td>
    <td>14.06</td>
    <td>18.48</td>
    <td>64.96</td>
    <td>89.78</td> -->
  </tr>
  <tr>
    <td align="left">AV-action</td> 
    <td>44.8/--</td>
    <!-- <td>15.86</td>
    <td>00.20</td>
    <td>03.66</td>
    <td>22.91</td>
    <td>73.08</td> -->
  </tr>
  <tr>
    <td align="left">UCF24 results</td> 
    <td></td>
    <!-- <td>31.80</td>
    <td>02.83</td>
    <td>11.42</td>
    <td>47.26</td>
    <td>85.49</td> -->
  </tr>
  <tr>
    <td align="left">Actionness</td> 
    <td>--</td>
    <!-- <td>39.95</td>
    <td>11.36</td>
    <td>17.47</td>
    <td>65.66</td>
    <td>89.78</td> -->
  </tr>
  <tr>
    <td align="left">Action detection</td> 
    <td>--</td>
    <!-- <td>42.08</td>
    <td>12.45</td>
    <td>18.40</td>
    <td>61.82</td>
    <td>90.55</td> -->
  </tr>
  <tr>
    <td align="left">ActionNess-framewise</td> 
    <td>--</td>
    <!-- <td>43.19</td>
    <td>13.05</td>
    <td>18.87</td>
    <td>64.35</td>
    <td>91.54</td> -->
  </tr>
</table>


##### Download pre-trained weights
- Currently, we provide the models from above table: 
    * trained weights are available from my [Google Drive](https://drive.google.com/drive/folders/1tOwQtQD3HWiTTp_ZgPCEWd4W-UKiglbt?usp=sharing)   
- These models can be used to reproduce above table which is almost same as in our [paper](https://arxiv.org/pdf/2102.11585.pdf) 

## Citation
If this work has been helpful in your research please cite following articles:

    @ARTICLE {singh2022road,
    author = {Singh, Gurkirt and Akrigg, Stephen and Di Maio, Manuele and Fontana, Valentina and Alitappeh, Reza Javanmard and Saha, Suman and Jeddisaravi, Kossar and Yousefi, Farzad and Culley, Jacob and Nicholson, Tom and others},
    journal = {IEEE Transactions on Pattern Analysis & Machine Intelligence},
    title = {ROAD: The ROad event Awareness Dataset for autonomous Driving},
    year = {5555},
    volume = {},
    number = {01},
    issn = {1939-3539},
    pages = {1-1},
    keywords = {roads;autonomous vehicles;task analysis;videos;benchmark testing;decision making;vehicle dynamics},
    doi = {10.1109/TPAMI.2022.3150906},
    publisher = {IEEE Computer Society},
    address = {Los Alamitos, CA, USA},
    month = {feb}
    }


    @inproceedings{singh2017online,
      title={Online real-time multiple spatiotemporal action localisation and prediction},
      author={Singh, Gurkirt and Saha, Suman and Sapienza, Michael and Torr, Philip HS and Cuzzolin, Fabio},
      booktitle={Proceedings of the IEEE International Conference on Computer Vision},
      pages={3637--3646},
      year={2017}
    }

    @article{maddern20171,
      title={1 year, 1000 km: The Oxford RobotCar dataset},
      author={Maddern, Will and Pascoe, Geoffrey and Linegar, Chris and Newman, Paul},
      journal={The International Journal of Robotics Research},
      volume={36},
      number={1},
      pages={3--15},
      year={2017},
      publisher={SAGE Publications Sage UK: London, England}
    }

