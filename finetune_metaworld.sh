#torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
# --vla_path "openvla/openvla-7b" \
#--data_root_dir ~/tensorflow_datasets \
#--dataset_name metaworld_ml10_50e \
#--run_root_dir /path/to/output \
#--adapter_tmp_dir /path/to/adapter-tmp \
#--lora_rank 32 \
#--batch_size 8 \
#--grad_accumulation_steps 2 \
#--learning_rate 5e-4 \
#--image_aug True \
#--max_steps 40000 \
#--save_steps 20000
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0


# 运行微调命令（所有 /Ao 全部替换为正确的绝对路径）
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "/root/autodl-tmp/openvla/openvla-7b" \
    --data_root_dir "/root/autodl-tmp/tensorflow_datasets" \
    --dataset_name metaworld_ml10_50e \
    --run_root_dir "/root/autodl-tmp/openvla/output" \
    --adapter_tmp_dir "/root/autodl-tmp/openvla/adapter-tmp" \
    --lora_rank 32 \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --image_aug True \
    --max_steps 20000 \
    --save_steps 10000 \
    --wandb_project "ml10" \
    --wandb_entity "1469512941-" \
    2>&1 | tee /root/autodl-tmp/openvla/output/train_log.txt
    #以上分别是微调代码和微调脚本
