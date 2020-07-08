## Devlopement
    [x] implement building action tubes in python given frame label detection based on ICCV17 paper
    [x] fill the gaps in genrated tubes;
    [x] Write evaluation protocal for tube of all types;
    [] fix detection scaling, i.e. normlise the output of network to [0-1]
    [] write frameAP evaluation All 6
    [] Generate tubes with marginals for duplexes and triplets
    [] add clssifcation for ego action classification for AV-actions
    [] frame-ap evaluation for AV-actions

## Fix annotations
    [x] integrate last video from reza, video annotate by alex bruce but correction not complete.
    [x] Redefine splits -  remove the video the shorter video from evaluation
    [x] integrate last video from Andy
    [] Fix start and end of the interploated videos @12fps from old version of VOTT

## Experimentst
    [] run experiment with different input sizes 
    [] run experiment with different network achtecture
    [x] see if we can accomodate YOLOv4
    [] check if giving information about anchor-location help in determining location better


python main.py --mode=train --min_size=600 --max_size=1000 --basenet=resnet50 --train_subsets=train_3 && python main.py --mode=train --min_size=400 --max_size=600 --basenet=resnet50 --train_subsets=train_3
    

# Major devlopments
    [] accomdate multiple frames as inputs 
    [] try accomdating YOLOv4