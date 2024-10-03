# Task 1
python tools/train_net.py --num-gpus 2 \
--dist-url 'tcp://127.0.0.1:50200' \
--resume \
--config-file ./configs/OWOD/t1/t1_train.yaml \
SOLVER.IMS_PER_BATCH 8 \
SOLVER.BASE_LR 0.01 \
OUTPUT_DIR "./output/t1"

# Task 2
# cp -r output/t1/* output/t2

# python tools/train_net.py --num-gpus 2 \
# --config-file ./configs/OWOD/t2/t2_train.yaml \
# SOLVER.IMS_PER_BATCH 8 \
# SOLVER.BASE_LR 0.01 \
# OUTPUT_DIR "./output/t2" \
# MODEL.WEIGHTS "output/t2/model_final.pth"