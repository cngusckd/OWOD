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
cp -r /home/joseph/workspace/OWOD/output/t1 /home/joseph/workspace/OWOD/output/t2

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/t2/t2_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t2" MODEL.WEIGHTS "/home/joseph/workspace/OWOD/output/t2/model_final.pth"

cp -r /home/joseph/workspace/OWOD/output/t2 /home/joseph/workspace/OWOD/output/t2_ft

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t2/t2_ft.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t2_ft" MODEL.WEIGHTS "/home/joseph/workspace/OWOD/output/t2_ft/model_final.pth"

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t2/t2_val.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "/home/joseph/workspace/OWOD/output/t2_ft/model_final.pth"

python tools/train_net.py --num-gpus 8 --eval-only --config-file ./configs/OWOD/t2/t2_test.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "/home/joseph/workspace/OWOD/output/t2_ft/model_final.pth"
