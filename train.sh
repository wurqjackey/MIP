# SYSU-MM01

python train.py --config_file configs/SYSU-MM/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')" OUTPUT_DIR 'logs/sysu_WHAT' SOLVER.BASE_LR 0.01 SOLVER.EVAL_PERIOD 1 INPUT.AUG 2 \
INPUT.SIZE_TRAIN [256,128] INPUT.SIZE_TEST [256,128] MODEL.MSEL True MODEL.MSEL_EPOCH 10 DATASETS.SAMPLER 'modal' MODEL.MSEL_MODAL True DATALOADER.NUM_INSTANCE 8 \
MODEL.USE_PROMPT True MODEL.NUM_TOKEN 16 MODEL.PROMPT_SCALE 20.0 MODEL.PROMPT_SHIFT 0.0 MODEL.USE_INS_PROMPT True MODEL.USE_INS_PROMPT_GEN True MODEL.NUM_INS_PMT_TOKEN 16 \
SOLVER.WEIGHT_IPLR 15.0 SOLVER.WEIGHT_MPLR 5.0 MODEL.IPIL False SOLVER.MAX_EPOCHS 80 TEST.EVAL_EPOCH 10 SOLVER.SCHEDULER 'cosine-refine' SOLVER.MIN_INDEX 0.00001 SOLVER.COSINE_EPOCHS 30
