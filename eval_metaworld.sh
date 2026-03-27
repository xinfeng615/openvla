
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

python experiments/robot/metaworld/run_metaworld_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /root/autodl-tmp/openvla/output/openvla-7b+metaworld_ml10_50e+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
  --task_suite_name metaworld_ml10_50e \
  --center_crop True\
  --use_wandb  True  \
  --wandb_project "ml10-eval" \
  --wandb_entity "1469512941-" 

#export CUDA_VISIBLE_DEVICES=0
#export MUJOCO_GL=egl

# 使用 xvfb-run -a 解决无头服务器的 EGL 渲染崩溃问题

#xvfb-run -a python experiments/robot/metaworld/run_metaworld_eval.py \
  #--model_family openvla \
  #--pretrained_checkpoint /root/autodl-tmp/openvla/openvla-7b \
  #--task_suite_name metaworld_ml10_50e \
  #--center_crop False
