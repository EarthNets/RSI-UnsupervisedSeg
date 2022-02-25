K_train=5
K_test=5
bsize=4
bsize_train=4
bsize_test=4
num_epoch=10
KM_INIT=20
KM_NUM=1
KM_ITER=20
SEED=1
LR=1e-4
num_workers=4

mkdir -p outputs/rn18/rgbd/mdc/test/${SEED}

python train_mdc.py \
--data_root /mnt/d/codes/datasets/unsup_seg/RGBD/ \
--save_root outputs/rn18/rgbd/mdc/test/${SEED}/ \
--eval_only \
--eval_path outputs/rn18/rgbd/mdc/train/1/scratch/augmented/res1=512_res2=512/jitter=True_blur=True_grey=True/equiv/h_flip=True_v_flip=False_crop=True/min_scale=0.5/K_train=5_cosine/checkpoint_1.pth.tar \
--arch resnet18 \
--in_dim 128 \
--repeats 1 \
--lr ${LR} \
--seed ${SEED} \
--num_init_batches ${KM_INIT} \
--num_batches ${KM_NUM} \
--kmeans_n_iter ${KM_ITER} \
--K_train ${K_train} --K_test ${K_test} \
--rgbd  \
--batch_size_cluster ${bsize}  \
--batch_size_train ${bsize_train} \
--batch_size_test ${bsize_test} \
--num_workers ${num_workers} \
--num_epoch ${num_epoch} \
--res 1024 --res1 1024 --res2 1024 \
--augment --jitter --blur --grey --equiv --random_crop --h_flip \
#--pretrain \
