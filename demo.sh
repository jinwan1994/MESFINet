#---training---
# BI, scale 2, 4
# MESFINet in the paper (x2)
# CUDA_VISIBLE_DEVICES=3,4 python train.py --model_name MESFINet --scale_factor 2 --checkpoints_dir ./log \
# --edge_model_path ./pretrained_model/bdcn_pretrained_on_nyudv2_rgb.pth 
#
# MESFINet in the paper (x4) - from MESFINet (x2)
# CUDA_VISIBLE_DEVICES=3,4 python train.py --model_name MESFINet --scale_factor 4 --checkpoints_dir ./log --edge_model_path ./pretrained_model/bdcn_pretrained_on_nyudv2_rgb.pth \
# --load_pretrain_modelx2 True --modelx2_path ./pretrained_model/MESFINet_2xSR_final.pth.tar
#

#---testing---
# BI, scale 2, 4
# Standard benchmarks (Ex. MESFINetx2)
# python test.py --model_name MESFINet_2xSR --scale_factor 2 --save_dir ./results_test --sr_model_path ./pretrained_model/MESFINet_2xSR_final.pth.tar \
# 	--dataset_list KITTI2012+KITTI2015+Middlebury --edge_model_path ./pretrained_model/bdcn_pretrained_on_nyudv2_rgb.pth
#
# Standard benchmarks (Ex. MESFINet_x4)
python test.py --model_name MESFINet_4xSR --scale_factor 4 --save_dir ./results_test --sr_model_path ./pretrained_model/MESFINet_4xSR_final.pth.tar \
	--dataset_list KITTI2012+KITTI2015+Middlebury --edge_model_path ./pretrained_model/bdcn_pretrained_on_nyudv2_rgb.pth