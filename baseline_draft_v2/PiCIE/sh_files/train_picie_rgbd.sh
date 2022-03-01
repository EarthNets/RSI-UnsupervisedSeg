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
LR=0.001
num_workers=4

mkdir -p outputs/rn18/rgbd/picie/train/${SEED}/

python train_picie.py \
--data_root /mnt/d/codes/datasets/unsup_seg/RGBD/ \
--save_root outputs/swin_t/rgbd/picie/train/${SEED}/ \
--arch swin_t \
--in_dim 256 \
--lr ${LR} \
--repeats 1 \
--seed ${SEED} \
--num_init_batches ${KM_INIT} \
--num_batches ${KM_NUM} \
--kmeans_n_iter ${KM_ITER} \
--K_train ${K_train} --K_test ${K_test} \
--batch_size_train ${bsize_train} \
--batch_size_test ${bsize_test} \
--batch_size_cluster ${bsize} \
--num_workers ${num_workers} \
--num_epoch ${num_epoch} \
--rgbd \
--res 512 --res1 512 --res2 512 \
--augment \
--grey \
--blur \
--jitter \
--equiv \
--random_crop \
--h_flip \
#--pretrain \