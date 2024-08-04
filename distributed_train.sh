torchrun --nproc_per_node=2 --nnodes=1 --master_port=44875 main_mt.py --batch_size 6 --epochs 100 --input_size 224 --blr 4e-4 --weight_decay 0.05 --warmup_epochs 10 --model mtvit_taskgate_att_mlp_base_MI_twice --drop_path 0.1 --scaleup --exp-name scaleup_mtvit_taskgate_att_mlp_base_MI_twice
