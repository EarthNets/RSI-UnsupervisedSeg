EVAL_PATH='outputs/rn18/rgb/picie/train/1/scratch/augmented/res1=512_res2=512/jitter=True_blur=True_grey=True/equiv/h_flip=True_v_flip=False_crop=True/min_scale=0.5/K_train=5_cosine/checkpoint_1.pth.tar' # Your checkpoint directory. 

mkdir -p outputs/rn18/rgb/picie/test/

python train_picie.py \
--data_root /mnt/d/codes/datasets/unsup_seg/RGBD/ \
--eval_only \
--save_root outputs/rn18/rgb/picie/test/ \
--eval_path ${EVAL_PATH} \
--arch resnet18 \
--in_dim 128 \
--res 1024 --res1 1024 --res2 1024 \
--rgb \
--K_test 5 \
--K_train 5 \
--batch_size_train 4 \
--batch_size_test 4 \
--batch_size_cluster 4 \
