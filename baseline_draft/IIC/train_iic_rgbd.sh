python -m code.scripts.segmentation.segmentation_twohead \
--mode IID \
--dataset RGBD \
--dataset_root /mnt/d/codes/datasets/unsup_seg/RGBD \
--out_root /mnt/d/codes/github/IIC/outputs/rgbd \
--model_ind 512 \
--arch SegmentationNetRN50TwoHead \
--num_epochs 4800 \
--lr 0.0001 \
--lamb_A 1.0 \
--lamb_B 1.0 \
--num_sub_heads 1 \
--batch_sz 4 \
--num_workers 4 \
--num_dataloaders 1 \
--output_k_A 15 \
--output_k_B 5 \
--gt_k 5 \
--input_sz 512 \
--half_T_side_sparse_min 0 \
--half_T_side_sparse_max 0 \
--half_T_side_dense 5 \
--include_rgb  \
--use_uncollapsed_loss \
--batchnorm_track \
--no_sobel \
--jitter_brightness 0.1 \
--jitter_contrast 0.1 \
--jitter_saturation 0.1 \
--jitter_hue 0.1 \
#--pre_scale_all \
#--pre_scale_factor 1.0 \