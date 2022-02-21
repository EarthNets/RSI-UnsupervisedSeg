EVAL_PATH='outputs/rgb/picie/train/checkpoint_2.pth.tar' # Your checkpoint directory. 

mkdir -p outputs/rgb/picie/test/

python train_picie.py \
--data_root /mnt/d/codes/datasets/unsup_seg/RGBD/ \
--eval_only \
--save_root outputs/rgbd/picie/test/ \
--eval_path ${EVAL_PATH} \
--arch resnet50 \
--in_dim 256 \
--res 1024 --res1 1024 --res2 1024 \
--rgb \
--K_test 5 \
--K_train 5 \
--batch_size_train 4 \
--batch_size_test 4 \
--batch_size_cluster 4 \
