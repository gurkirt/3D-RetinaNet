# #DONE python main.py --MODE=train --MODEL_TYPE=C2D --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=I3D --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=C2D --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=I3D --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=C2D --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=C2D --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=I3D --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=16 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3 --MIN_SIZE=512
# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=16 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3 --MIN_SIZE=512
# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=16 --CLS_HEAD_TIME_SIZE=3 
# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 
# python main.py --MODE=train --MODEL_TYPE=RCLSTM --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=I3D --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=C2D-NL --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=I3D-NL --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=RCN-NL --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=C2D --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=C2D-NL --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=gen_dets --MODEL_TYPE=RCN --NMS_THRESH=0.55 --TOPK=2000 
# python main.py --MODE=gen_dets --MODEL_TYPE=C2D --NMS_THRESH=0.55 --TOPK=2000 
# python main.py --MODE=gen_dets --MODEL_TYPE=I3D --NMS_THRESH=0.55 --TOPK=2000 

CUDA_VISIBLE_DEVICES=1 python main.py --MODE=gen_dets --MODEL_TYPE=I3D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=1 python main.py --MODE=gen_dets --MODEL_TYPE=I3D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4\
&& CUDA_VISIBLE_DEVICES=1 python main.py --MODE=gen_dets --MODEL_TYPE=I3D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4 
CUDA_VISIBLE_DEVICES=2 python main.py --MODE=gen_dets --MODEL_TYPE=RCN --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=2 python main.py --MODE=gen_dets --MODEL_TYPE=RCN --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=2 python main.py --MODE=gen_dets --MODEL_TYPE=RCN --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4 

CUDA_VISIBLE_DEVICES=3 python main.py --MODE=gen_dets --MODEL_TYPE=C2D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=3 python main.py --MODE=gen_dets --MODEL_TYPE=C2D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=2 python main.py --MODE=gen_dets --MODEL_TYPE=C2D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4 

CUDA_VISIBLE_DEVICES=0 python main.py --MODE=gen_dets --MODEL_TYPE=I3D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=0 python main.py --MODE=gen_dets --MODEL_TYPE=I3D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4\
&& CUDA_VISIBLE_DEVICES=0 python main.py --MODE=gen_dets --MODEL_TYPE=I3D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 

CUDA_VISIBLE_DEVICES=2 python main.py --MODE=gen_dets --MODEL_TYPE=RCN --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=2 python main.py --MODE=gen_dets --MODEL_TYPE=RCN --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=2 python main.py --MODE=gen_dets --MODEL_TYPE=RCN --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 

CUDA_VISIBLE_DEVICES=3 python main.py --MODE=gen_dets --MODEL_TYPE=C2D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=3 python main.py --MODE=gen_dets --MODEL_TYPE=C2D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=3 python main.py --MODE=gen_dets --MODEL_TYPE=C2D --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4
CUDA_VISIBLE_DEVICES=0 python main.py --MODE=gen_dets --MODEL_TYPE=RCLSTM --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=0 python main.py --MODE=gen_dets --MODEL_TYPE=RCLSTM --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4 \
&& CUDA_VISIBLE_DEVICES=0 python main.py --MODE=gen_dets --MODEL_TYPE=RCLSTM --NMS_THRESH=0.5 --TEST_BATCH_SIZE=1 --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4


58.215075731277466,78.31317782402039, 70.03562450408936, 0.28206510469317436, 44.170862436294556, 10.025178641080856,21.333950757980347,0.0,65.36361575126648,71.10176086425781

agent_ness : 82465 : 330748 : 70.15333771705627                        
[INFO: evaluation.py:  425]: Ped : 30217 : 330748 : 58.215075731277466                                                                                                             
[INFO: evaluation.py:  425]: Car : 17029 : 330748 : 78.31317782402039                                                                                                            
[INFO: evaluation.py:  425]: Cyc : 10587 : 330748 : 70.03562450408936                                                                                                              
[INFO: evaluation.py:  425]: Mobike : 25 : 330748 : 0.28206510469317436                              
[INFO: evaluation.py:  425]: MedVeh : 4246 : 330748 : 44.170862436294556                                                                                                           
[INFO: evaluation.py:  425]: LarVeh : 3341 : 330748 : 10.025178641080856            
[INFO: evaluation.py:  425]: Bus : 831 : 330748 : 21.333950757980347                                                                                                               
[INFO: evaluation.py:  425]: EmVeh : 1 : 330748 : 0.0                                  
[INFO: evaluation.py:  425]: TL : 14180 : 330748 : 65.36361575126648                                                                                                               
[INFO: evaluation.py:  425]: OthTL : 2009 : 330748 : 71.10176086425781                                                                                                            
[INFO: evaluation.py:  425]: Red : 11159 : 330748 : 56.04601502418518                                                                                                            
[INFO: evaluation.py:  425]: Amber : 976 : 330748 : 22.728580236434937                                                                                                             
[INFO: evaluation.py:  425]: Green : 4014 : 330748 : 33.983346819877625                                                                                                          
[INFO: evaluation.py:  425]: MovAway : 22143 : 330748 : 46.43758237361908                                                                                                          
[INFO: evaluation.py:  425]: MovTow : 17628 : 330748 : 49.557772278785706                                                                                                       
[INFO: evaluation.py:  425]: Mov : 2527 : 330748 : 35.52559018135071                                                                                                               
[INFO: evaluation.py:  425]: Brake : 2757 : 330748 : 26.682499051094055                                                                                                           
[INFO: evaluation.py:  425]: Stop : 17215 : 330748 : 35.89448630809784                                                                                                          
[INFO: evaluation.py:  425]: IncatLft : 1810 : 330748 : 8.978598564863205                                                                                                          
[INFO: evaluation.py:  425]: IncatRht : 706 : 330748 : 2.6991231366991997                            
[INFO: evaluation.py:  425]: HazLit : 682 : 330748 : 3.284970670938492                                                                                                             
[INFO: evaluation.py:  425]: TurLft : 816 : 330748 : 3.8237884640693665                                                                                                            
[INFO: evaluation.py:  425]: TurRht : 984 : 330748 : 5.627048015594482     

# python main.py --MODE=train --MODEL_TYPE=RCLSTM
# python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TRAIN_SUBSETS=train_2
# python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TRAIN_SUBSETS=train_2
# python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TRAIN_SUBSETS=train_2
# python main.py --MODE=train --MODEL_TYPE=RCLSTM --TRAIN_SUBSETS=train_2
# python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TRAIN_SUBSETS=train_1 
# python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TRAIN_SUBSETS=train_1
# python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TRAIN_SUBSETS=train_1
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --MODE=train --MODEL_TYPE=RCLSTM --TRAIN_SUBSETS=train_1
# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=16 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3 --MIN_SIZE=512


CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --MODE=train --MODEL_TYPE=RCN --DATASET=ucf24 --VAL_SUBSETS=val --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --MODE=train --MODEL_TYPE=RCLSTM --DATASET=ucf24 --VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10

python main.py --MODE=train --MODEL_TYPE=C2D --DATASET=ucf24 --VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10