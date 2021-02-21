# 3D-RetinaNet for ROAD and UCF-24 dataset
This repository contains code for 3D-RetinaNet, which is used a baseline for [ROAD dataset](https://github.com/gurkirt/road-dataset) in dataset release [paper](). It contains training and evaluation for ROAD and UCF-24 datasets. 



## Table of Contents
- <a href='#Requirements'>Requirements</a>
- <a href='#training'>Training 3D-RetinaNet</a>
- <a href='#evaluation'>Building Tubes</a>
- <a href='#performance'>Performance</a>
- <a href='#online-code'>Online-Code</a>
- <a href='#extras'>Extras</a>
- <a href='#todo'>TODO</a>
- <a href='#citation'>Citation</a>
- <a href='#references'>Reference</a>


## Requirements
We need three things to get started with training: datasets, kinetics pre-trained weight, and pytorch with torchvision and tensoboardX. 

### Dataset download an pre-process

- We currently only support following two dataset.
    - [ROAD dataset](https://github.com/gurkirt/road-dataset) in dataset release [paper]()
    - [UCF24](http://www.thumos.info/download.html) with [revised annotations](https://github.com/gurkirt/corrected-UCF101-Annots) released with our [ICCV-2017 paper](https://arxiv.org/pdf/1611.08563.pdf).

- Visit [ROAD dataset](https://github.com/gurkirt/road-dataset) for download and pre-processing. 
- You can download `rgb-images` it from my [google drive link](https://drive.google.com/file/d/1o2l6nYhd-0DDXGP-IPReBP4y1ffVmGSE/view?usp=sharing) for UCF24 Dataset. Download annotations from [corrected-UCF10-annots-repo](https://github.com/gurkirt/corrected-UCF101-Annots/blob/master/pyannot_with_class_names.pkl). 
    - Your data directory should look like:
        ```
        - ucf24/
            - pyannot_with_class_names.pkl
            - rgb-images
                - video-name ...
                    - images ......
        ```
### Pytorch and weights
    - Install [Pytorch](https://pytorch.org/) and [torchvision](http://pytorch.org/docs/torchvision/datasets.html)
    - TensorboardX `pip install tensorboardx`
    - Pre-trained weight on [kinetics-400](https://deepmind.com/research/open-source/kinetics). Download them from from [Google-Drive](https://drive.google.com/drive/folders/1tOwQtQD3HWiTTp_ZgPCEWd4W-UKiglbt?usp=sharing). Name the folder `kinetics-pt`, it is important to name it right. 

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

- Note:
  * Network occupies almost 9.7GB VRAM on each GPU, we used 1080Ti for training and normal training takes about 24 hrs on road dataset.
  * During training checkpoint is saved every epoch also log it's frame-level `frame-mean-ap` on a subset of validation split test.


## Testing and Building Tubes
To generate the tubes and evaluate them, first, you will need frame-level detection and link them. It is pretty simple in out case. Similar to training command, you can run following commands. These can run on single GPUs. 


##### produce frame-level detection
Once you have trained network then you can use `test-ucf24.py` to generate frame-level detections.
To eval SSD using the test script simply specify the parameters listed in `test-ucf24.py` as a flag or manually change them. for e.g.:
```Shell
CUDA_VISIBLE_DEVICES=0 python3 test-ucf24.py --data_root=/home/user/ucf24/ --save_root=/home/user/ucf24/
--input_type=rgb --eval_iter=120000
```

To evaluate on optical flow models

```Shell
CUDA_VISIBLE_DEVICES=0 python3 test-ucf24.py --data_root=/home/user/ucf24/ --save_root=/home/user/ucf24/
--input_type=brox --eval_iter=120000
```

-Note
  * By default it will compute frame-level detections and store them as well as compute frame-mean-AP in models saved at 90k and 120k iteration.
  * There is a log file created for each iteration's frame-level evaluation.

##### Build tubes
You will need frame-level detections and you will need to navigate to `online-tubes`

Step-1: you will need to spacify `data_root`, `data_root` and `iteration_num_*` in `I01onlineTubes` and `I02genFusedTubes`;
<br>
Step 2: run  `I01onlineTubes` and `I02genFusedTubes` in matlab this print out video-mean-ap and save the results in a `.mat` file

Results are saved in `save_root/results.mat`. Additionally,`action-path` and `action-tubes` are also stroed under `save_root\ucf24\*` folders.

* NOTE: `I01onlineTubes` and `I02genFusedTubes` not only produce video-level mAP; they also produce video-level classification accuracy on 24 classes of UCF24.
##### frame-meanAP
To compute frame-mAP you can use `frameAP.m` script. You will need to specify `data_root`, `data_root`.
Use this script to produce results for your publication not the python one, both are almost identical,
but their ap computation from precision and recall is slightly different.

## Performance
##### UCF24 Test
The table below is similar to [table 1 in our paper](https://arxiv.org/pdf/1611.08563.pdf). It contains more info than
that in the paper, mostly about this implementation.
<table style="width:100% th">
  <tr>
    <td>IoU Threshold = </td>
    <td>0.20</td> 
    <td>0.50</td>
    <td>0.75</td>
    <td>0.5:0.95</td>
    <td>frame-mAP@0.5</td>
    <td>accuracy(%)</td>
  </tr>
  <tr>
    <td align="left">Peng et al [3] RGB+BroxFLOW </td> 
    <td>73.67</td>
    <td>32.07</td>
    <td>00.85</td> 
    <td>07.26</td>
    <td> -- </td> 
    <td> -- </td>
  </tr>
  <tr>
    <td align="left">Saha et al [2] RGB+BroxFLOW </td> 
    <td>66.55</td>
    <td>36.37</td> 
    <td>07.94</td>
    <td>14.37</td>
    <td> -- </td>
    <td> -- </td>
  </tr>
  <tr>
    <td align="left">Singh et al [4] RGB+FastFLOW </td> 
    <td>70.20</td>
    <td>43.00</td> 
    <td>14.10</td>
    <td>19.20</td>
    <td> -- </td>
    <td> -- </td>
  </tr>
  <tr>
    <td align="left">Singh et al [4] RGB+BroxFLOW </td> 
    <td>73.50</td>
    <td>46.30</td>
    <td>15.00</td> 
    <td>20.40</td>
    <td> -- </td>
    <td> 91.12 </td>  
  </tr>
  <tr>
    <td align="left">This implentation[4] RGB </td> 
    <td>72.08</td>
    <td>40.59</td>
    <td>14.06</td>
    <td>18.48</td>
    <td>64.96</td>
    <td>89.78</td>
  </tr>
  <tr>
    <td align="left">This implentation[4] FastFLOW </td> 
    <td>46.32</td>
    <td>15.86</td>
    <td>00.20</td>
    <td>03.66</td>
    <td>22.91</td>
    <td>73.08</td>
  </tr>
  <tr>
    <td align="left">This implentation[4] BroxFLOW </td> 
    <td>68.33</td>
    <td>31.80</td>
    <td>02.83</td>
    <td>11.42</td>
    <td>47.26</td>
    <td>85.49</td>
  </tr>
  <tr>
    <td align="left">This implentation[4] RGB+FastFLOW (boost-fusion) </td> 
    <td>71.38</td>
    <td>39.95</td>
    <td>11.36</td>
    <td>17.47</td>
    <td>65.66</td>
    <td>89.78</td>
  </tr>
  <tr>
    <td align="left">This implentation[4] RGB+FastFLOW (union-set) </td> 
    <td>73.68</td>
    <td>42.08</td>
    <td>12.45</td>
    <td>18.40</td>
    <td>61.82</td>
    <td>90.55</td>
  </tr>
  <tr>
    <td align="left">This implentation[4] RGB+FastFLOW(mean fusion) </td> 
    <td>75.48</td>
    <td>43.19</td>
    <td>13.05</td>
    <td>18.87</td>
    <td>64.35</td>
    <td>91.54</td>
  </tr>
  <tr>
    <td align="left">This implentation[4] RGB+BroxFLOW (boost-fusion) </td> 
    <td>73.34</td>
    <td>42.47</td>
    <td>12.23</td>
    <td>18.67</td>
    <td>68.31</td>
    <td>90.88</td>
  </tr>
  <tr>
    <td align="left">This implentation[4] RGB+BroxFLOW (union-set) </td> 
    <td>75.01</td>
    <td>44.98</td>
    <td>13.89</td>
    <td>19.76</td>
    <td>64.97</td>
    <td>90.77</td>
  </tr>
  <tr>
    <td align="left">This implentation[4] RGB+BroxFLOW(mean fusion) </td> 
    <td>76.43</td>
    <td>45.18</td>
    <td>14.39</td>
    <td>20.08</td>
    <td>67.81</td>
    <td>92.20</td>
  </tr>
  <tr>
    <td align="left">Kalogeiton et al. [5] RGB+BroxFLOW (stack of flow images)(mean fusion) </td>
    <td>76.50</td>
    <td>49.20</td>
    <td>19.70</td>
    <td>23.40</td>
    <td>69.50</td>
    <td>--</td>
  </tr>
</table>

##### Discussion:
`Effect of training iterations:`
There is an effect due to the choice of learning rate and the number of iterations the model is trained.
If you train the SSD network on initial learning rate for
many iterations then it performs is better on
lower IoU threshold, which is done in this case.
In original work using caffe implementation of SSD,
I trained the SSD network  with 0.0005 learning rate for first 30K
iterations and dropped then learning rate by the factor of 5
(divided by 5) and further trained up to 45k iterations.
In this implementation, all the models are trained for 120K
iterations, the initial learning rate is set to 0.0005 and learning is dropped by the factor of 5 after 70K and 90K iterations.

`Kalogeiton et al. [5] ` make use mean fusion, so I thought we could try in our pipeline which was very easy to incorporate.
It is evident from above table that mean fusion performs better than other fusion techniques.
Also, their method relies on multiple frames as input in addition to post-processing of bounding box coordinates at tubelet level.

##### Real-time aspect:

This implementation is mainly focused on producing the best numbers (mAP) in the simplest manner, it can be modified to run faster.
There few aspect that would need changes:
 - NMS is performed once in python then again in Matlab; one has to do that on GPU in python
 - Most of the time spent during tube generations is taken by disc operations; which can be eliminated completely.
 - IoU computation during action path is done multiple time just to keep the code clean that can be handled more smartly

Contact me if you want to implement the real-time version.
The Proper real-time version would require converting Matlab part into python.
I presented the timing of individual components in the paper, which still holds true.

## Extras
To use pre-trained model download the pre-trained weights from the links given below and make changes in `test-ucf24.py` to accept the downloaded weights. 

##### Download pre-trained weights
- Currently, we provide the following PyTorch models: 
    * SSD300 trained on ucf24 ; available from my [google drive](https://drive.google.com/drive/folders/1Z42S8fQt4Amp1HsqyBOoHBtgVKUzJuJ8?usp=sharing)
      - appearence model trained on rgb-images (named `rgb-ssd300_ucf24_120000`)
      - accurate flow model trained on brox-images (named `brox-ssd300_ucf24_120000`)
      - real-time flow model trained on fastOF-images (named `fastOF-ssd300_ucf24_120000`)    
- These models can be used to reproduce above table which is almost identical in our [paper](https://arxiv.org/pdf/1611.08563.pdf) 

## TODO
 - Incorporate JHMDB-21 dataset
 - Convert matlab part into python (happy to accept PR)

## Citation
If this work has been helpful in your research please consider citing [1] and [4]

      @inproceedings{singh2016online,
        title={Online Real time Multiple Spatiotemporal Action Localisation and Prediction},
        author={Singh, Gurkirt and Saha, Suman and Sapienza, Michael and Torr, Philip and Cuzzolin, Fabio},
        jbooktitle={ICCV},
        year={2017}
      }

## References
- [1] Wei Liu, et al. SSD: Single Shot MultiBox Detector. [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [2] S. Saha, G. Singh, M. Sapienza, P. H. S. Torr, and F. Cuzzolin, Deep learning for detecting multiple space-time action tubes in videos. BMVC 2016 
- [3] X. Peng and C. Schmid. Multi-region two-stream R-CNN for action detection. ECCV 2016
- [4] G. Singh, S Saha, M. Sapienza, P. H. S. Torr and F Cuzzolin. Online Real time Multiple Spatiotemporal Action Localisation and Prediction. ICCV, 2017.
- [5] Kalogeiton, V., Weinzaepfel, P., Ferrari, V. and Schmid, C., 2017. Action Tubelet Detector for Spatio-Temporal Action Localization. ICCV, 2017.
- [Original SSD Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thanks to Max deGroot, Ellis Brown for Pytorch implementation of [SSD](https://github.com/amdegroot/ssd.pytorch)
 

