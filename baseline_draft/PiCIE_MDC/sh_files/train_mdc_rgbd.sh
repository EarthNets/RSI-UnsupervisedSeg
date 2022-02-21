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

mkdir -p outputs/rgbd/mdc/train/${SEED}

python train_mdc.py \
--data_root /mnt/d/codes/datasets/unsup_seg/RGBD/ \
--save_root outputs/rgbd/mdc/train/${SEED}/ \
--arch resnet50 \
--in_dim 256 \
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
--res 512 --res1 512 --res2 512 \
--augment --jitter --blur --grey --equiv --random_crop --h_flip 
