# #DONE python main.py --MODE=train --MODEL_TYPE=C2D --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=I3D --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=C2D --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=I3D --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=C2D --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=C2D --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=I3D --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3

python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3
python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3
python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=16 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3
python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3 --MIN_SIZE=512
python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=16 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3 --MIN_SIZE=512

# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=16 --CLS_HEAD_TIME_SIZE=3 
# python main.py --MODE=train --MODEL_TYPE=RCN --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 
# python main.py --MODE=train --MODEL_TYPE=RCLSTM --MODEL_PATH=kinetics --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=I3D --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=C2D-NL --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=I3D-NL --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1
# python main.py --MODE=train --MODEL_TYPE=RCN-NL --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=C2D --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=3 --REG_HEAD_TIME_SIZE=3
# python main.py --MODE=train --MODEL_TYPE=C2D-NL --MODEL_PATH=imagenet --SEQ_LEN=8 --CLS_HEAD_TIME_SIZE=1