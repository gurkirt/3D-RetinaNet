python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_1 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_1 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_2 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_2 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4
python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4
python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_3 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_3 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4
python main.py --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4
python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_1 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_1 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_2 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_2 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4
python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4
python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_3 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_3 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4
python main.py --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4
python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_1 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_1 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_2 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_2 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4
python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_2 --SEQ_LEN=8 --BATCH_SIZE=4
python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=16 --TRAIN_SUBSETS=train_3 --SEQ_LEN=16 --BATCH_SIZE=8
python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_3 --SEQ_LEN=16 --BATCH_SIZE=8

CUDA_VISIBLE_DEVICES=0 python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_1 --SEQ_LEN=8 --BATCH_SIZE=4
CUDA_VISIBLE_DEVICES=1 python main.py --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_1 --SEQ_LEN=8 --BATCH_SIZE=4
CUDA_VISIBLE_DEVICES=2 python main.py --MODE=gen_dets --MODEL_TYPE=RCLSTM --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4
CUDA_VISIBLE_DEVICES=3 python main.py --MODE=gen_dets --MODEL_TYPE=RCLSTM --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train_3 --SEQ_LEN=8 --BATCH_SIZE=4
python main.py --MODE=train --MODEL_TYPE=RCLSTM --DATASET=ucf24 --VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10 --VAL_STEP=1

CUDA_VISIBLE_DEVICES=0 python main.py --MODE=gen_dets --MODEL_TYPE=I3D --DATASET=ucf24 \
--VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TEST_SEQ_LEN=8 \
--LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10 --VAL_STEP=1 --EVAL_EPOCHS=10 && \
CUDA_VISIBLE_DEVICES=1 python main.py --MODE=gen_dets --MODEL_TYPE=I3D --DATASET=ucf24 \
--VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TEST_SEQ_LEN=32 \
--LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10 --VAL_STEP=1 --EVAL_EPOCHS=10

CUDA_VISIBLE_DEVICES=0 python main.py --MODE=gen_dets --MODEL_TYPE=C2D --DATASET=ucf24 \
--VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TEST_SEQ_LEN=8 \
--LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10 --VAL_STEP=1 --EVAL_EPOCHS=10 && \
CUDA_VISIBLE_DEVICES=1 python main.py --MODE=gen_dets --MODEL_TYPE=C2D --DATASET=ucf24 \
--VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TEST_SEQ_LEN=32 \
--LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10 --VAL_STEP=1 --EVAL_EPOCHS=10


CUDA_VISIBLE_DEVICES=2 python main.py --MODE=gen_dets --MODEL_TYPE=RCLSTM --DATASET=ucf24 \
--VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TEST_SEQ_LEN=8 \
--LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10 --VAL_STEP=1 --EVAL_EPOCHS=10 && \
CUDA_VISIBLE_DEVICES=3 python main.py --MODE=gen_dets --MODEL_TYPE=RCLSTM --DATASET=ucf24 \
--VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TEST_SEQ_LEN=32 \
--LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10 --VAL_STEP=1 --EVAL_EPOCHS=10

CUDA_VISIBLE_DEVICES=3 python main.py --MODE=gen_dets --MODEL_TYPE=RCN --DATASET=ucf24 \
--VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TEST_SEQ_LEN=8 \
--LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10 --VAL_STEP=1 --EVAL_EPOCHS=10 && \
CUDA_VISIBLE_DEVICES=2 python main.py --MODE=gen_dets --MODEL_TYPE=RCN --DATASET=ucf24 \
--VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TEST_SEQ_LEN=32 \
--LR=0.00245 --MILESTONES=6,8 --MAX_EPOCHS=10 --VAL_STEP=1 --EVAL_EPOCHS=10


python main.py --MODE=eval_tubes --MODEL_TYPE=I3D --DATASET=ucf24 --VAL_SUBSETS=test --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TEST_SEQ_LEN=8 --EVAL_EPOCHS=10 --COMPUTE_PATHS=True --COMPUTE_TUBES=True  --NMS_THRESH=0.85

|Basketball : 35 : 898 : 1.2602046132087708                                                                                                                 
|BasketballDunk : 38 : 641 : 15.338398516178131                                                                                                             
|Biking : 67 : 1049 : 61.04176640510559                                                                                                                     
|CliffDiving : 40 : 861 : 47.2550243139267                                                                                                                  
|CricketBowling : 38 : 477 : 3.4921307116746902                                                                                                             
|Diving : 45 : 1109 : 67.5191879272461                                                                                                                      
|Fencing : 81 : 837 : 88.49281072616577                                                                                                                     
|FloorGymnastics : 36 : 860 : 96.80678844451904                                                                                                             
|GolfSwing : 39 : 495 : 34.27730202674866                                       
|HorseRiding : 51 : 1108 : 92.52873063087463                                                                                                                
|IceDancing : 107 : 953 : 75.48564076423645                                                                  
|LongJump : 38 : 721 : 67.27762222290039                                        
|PoleVault : 43 : 1462 : 68.82388591766357                                      
|RopeClimbing : 34 : 920 : 98.66253137588501                                                                                                                
|SalsaSpin : 210 : 975 : 5.712602287530899                                                                                                                  
|SkateBoarding : 32 : 1008 : 96.875                                             
|Skiing : 40 : 1040 : 79.83641028404236                                                                                                                    
|Skijet : 29 : 1201 : 74.50543642044067                                                                                                                     
|SoccerJuggling : 41 : 1007 : 86.44051551818848                                 
|Surfing : 52 : 990 : 56.359678506851196                                                   
|TennisSwing : 66 : 826 : 0.584816001355648                                     
|TrampolineJumping : 76 : 853 : 38.90399634838104                               
|VolleyballSpiking : 37 : 832 : 1.5950735658407211                              
|WalkingWithDog : 40 : 918 : 56.385356187820435




CUDA_VISIBLE_DEVICES=1 python main.py --DATASET=ucf24 --VAL_SUBSETS=test --EVAL_EPOCHS=10 --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TOPK=15 --CONF_THRESH=0.025 --GEN_NMS=0.8 --PATHS_IOUTH=0.25 --PATHS_COST_TYPE=score --PATHS_JUMP_GAP=4 --TOPK=20 --PATHS_MIN_LEN=4 --TRIM_METHOD=indiv --TUBES_ALPHA=0 && CUDA_VISIBLE_DEVICES=1 python main.py --DATASET=ucf24 --VAL_SUBSETS=test --EVAL_EPOCHS=10 --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TOPK=15 --CONF_THRESH=0.025 --GEN_NMS=0.8 --PATHS_IOUTH=0.25 --PATHS_COST_TYPE=score --PATHS_JUMP_GAP=4 --TOPK=20 --PATHS_MIN_LEN=4 --TRIM_METHOD=indiv --TUBES_ALPHA=0 && CUDA_VISIBLE_DEVICES=1 python main.py --DATASET=ucf24 --VAL_SUBSETS=test --EVAL_EPOCHS=10 --MODE=gen_dets --MODEL_TYPE=I3D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TOPK=15 --CONF_THRESH=0.025 --GEN_NMS=0.8 --PATHS_IOUTH=0.25 --PATHS_COST_TYPE=score --PATHS_JUMP_GAP=4 --TOPK=20 --PATHS_MIN_LEN=4 --TRIM_METHOD=indiv --TUBES_ALPHA=0

CUDA_VISIBLE_DEVICES=3 python main.py --DATASET=ucf24 --VAL_SUBSETS=test --EVAL_EPOCHS=10 --MODE=gen_dets --MODEL_TYPE=C2D --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TOPK=15 --CONF_THRESH=0.025 --GEN_NMS=0.8 --PATHS_IOUTH=0.25 --PATHS_COST_TYPE=score --PATHS_JUMP_GAP=4 --TOPK=20 --PATHS_MIN_LEN=4 --TRIM_METHOD=indiv --TUBES_ALPHA=0 && CUDA_VISIBLE_DEVICES=3 python main.py --DATASET=ucf24 --VAL_SUBSETS=test --EVAL_EPOCHS=10 --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TOPK=15 --CONF_THRESH=0.025 --GEN_NMS=0.8 --PATHS_IOUTH=0.25 --PATHS_COST_TYPE=score --PATHS_JUMP_GAP=4 --TOPK=20 --PATHS_MIN_LEN=4 --TRIM_METHOD=indiv --TUBES_ALPHA=0

CUDA_VISIBLE_DEVICES=2 python main.py --DATASET=ucf24 --VAL_SUBSETS=test --EVAL_EPOCHS=10 --MODE=gen_dets --MODEL_TYPE=RCN --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TOPK=15 --CONF_THRESH=0.025 --GEN_NMS=0.8 --PATHS_IOUTH=0.25 --PATHS_COST_TYPE=score --PATHS_JUMP_GAP=4 --TOPK=20 --PATHS_MIN_LEN=4 --TRIM_METHOD=indiv --TUBES_ALPHA=0 && CUDA_VISIBLE_DEVICES=2 python main.py --DATASET=ucf24 --VAL_SUBSETS=test --EVAL_EPOCHS=10 --MODE=gen_dets --MODEL_TYPE=RCLSTM --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TOPK=15 --CONF_THRESH=0.025 --GEN_NMS=0.8 --PATHS_IOUTH=0.25 --PATHS_COST_TYPE=score --PATHS_JUMP_GAP=4 --TOPK=20 --PATHS_MIN_LEN=4 --TRIM_METHOD=indiv --TUBES_ALPHA=0 && CUDA_VISIBLE_DEVICES=2 python main.py --DATASET=ucf24 --VAL_SUBSETS=test --EVAL_EPOCHS=10 --MODE=gen_dets --MODEL_TYPE=RCLSTM --TEST_SEQ_LEN=8 --TRAIN_SUBSETS=train --SEQ_LEN=8 --BATCH_SIZE=4 --TOPK=15 --CONF_THRESH=0.025 --GEN_NMS=0.8 --PATHS_IOUTH=0.25 --PATHS_COST_TYPE=score --PATHS_JUMP_GAP=4 --TOPK=20 --PATHS_MIN_LEN=4 --TRIM_METHOD=indiv --TUBES_ALPHA=0

THS=False --COMPUTE_TUBES=True --DATASET=ucf24 --VAL_SUBSETS=test --EVAL_EPOCHS=10 --MODE=eval_tubes --MODEL_TYPE=I3D --TEST_SEQ_LEN=32 --TRAIN_SUBSETS=
train --SEQ_LEN=8 --BATCH_SIZE=4 --GEN_NMS=0.80 --PATHS_IOUTH=0.25 --PATHS_COST_TYPE=score --PATHS_JUMP_GAP=4 --TOPK=20 --PATHS_MIN_LEN=4 --TRIM_METHOD=indiv --TUBES_ALPHA=0