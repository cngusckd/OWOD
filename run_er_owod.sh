python tools/train_er_owod.py --num-gpus 2 \
--dist-url 'tcp://127.0.0.1:50302' \
--resume \
--config-file ./configs/OWOD/t2/t2_train.yaml \
SOLVER.IMS_PER_BATCH 2 \
SOLVER.BASE_LR 0.01 \
MODEL.WEIGHTS "output/t1/model_0004999.pth" \
OUTPUT_DIR "./owod_er_output/t2"