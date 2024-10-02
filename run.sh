# Task 1
python tools/train_net.py --num-gpus 2 \
--resume \
--config-file ./configs/OWOD/t1/t1_train.yaml \
SOLVER.IMS_PER_BATCH 8 \
SOLVER.BASE_LR 0.01 \
OUTPUT_DIR "./output/t1"

# No need to finetune in Task 1, as there is no incremental component.

# python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t1/t1_val.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "/home/joseph/workspace/OWOD/output/t1/model_final.pth"

# python tools/train_net.py --num-gpus 8 --eval-only --config-file ./configs/OWOD/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "/home/joseph/workspace/OWOD/output/t1/model_final.pth"


# Task 2
cp -r output/t1/* output/t2

python tools/train_net.py --num-gpus 2 \
--config-file ./configs/OWOD/t2/t2_train.yaml \
SOLVER.IMS_PER_BATCH 8 \
SOLVER.BASE_LR 0.01 \
OUTPUT_DIR "./output/t2" \
MODEL.WEIGHTS "output/t2/model_final.pth"
