python -m code.scripts.segmentation.segmentation_twohead \
--mode IID --dataset Potsdam --dataset_root /mnt/d/codes/datasets/unsup_seg/POTSDAM --model_ind 544 --arch SegmentationNet10aTwoHead --num_epochs 4800 --lr 0.000001 --lamb_A 1.0 --lamb_B 1.0 --num_sub_heads 1 --batch_sz 16 --num_dataloaders 1 --output_k_A 36 --output_k_B 6 --gt_k 6 --input_sz 200 --half_T_side_sparse_min 0 --half_T_side_sparse_max 0 --half_T_side_dense 5  --include_rgb --no_sobel --jitter_brightness 0.1 --jitter_contrast 0.1 --jitter_saturation 0.1 --jitter_hue 0.1 --use_uncollapsed_loss --batchnorm_track