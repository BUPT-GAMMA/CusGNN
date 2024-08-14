#!/usr/bin/env bash
(
gpu=0
nl=3
exp_id=""$gpu"_gpu_"$nl""
python train_search.py --max_epochs 100 --hidden_size 128 --batch_size 8192 --catalog_name catA --gpu $gpu --num_layers $nl  --model_path catA --exp_id $exp_id --save_searched_file "searched_arch_"$exp_id""  --seed 0 2>&1 >> logs/"catA_search_"$exp_id".log"
python train_sub_network.py --hidden_size 2048 --learning_rate 0.0001 --learning_rate_min 0.0001 --weight_decay 0.0001 --num_layers $nl  --max_epochs 100 --skip_training False --exp_id $exp_id --save_searched_file "searched_arch_"$exp_id""  --catalog_name catA  --model_path catA --gpu $gpu --seed 0 2>&1 >> logs/catA_fine_"$exp_id".log
)&

(
gpu=2
nl=3
exp_id=""$gpu"_gpu_"$nl""
python train_search.py --max_epochs 20 --hidden_size 128 --batch_size 8192 --catalog_name catC --gpu $gpu --model_path catC_read --num_layers $nl  --exp_id $exp_id --save_searched_file "searched_arch_"$exp_id"" --seed 0 2>&1 >> logs/"catC_search_"$exp_id".log";
python train_sub_network.py --hidden_size 2048 --learning_rate 0.0001 --learning_rate_min 0.0001 --weight_decay 0.0001 --num_layers $nl  --max_epochs 20 --skip_training False --exp_id $exp_id --save_searched_file "searched_arch_"$exp_id""  --catalog_name catC  --model_path catC --gpu $gpu --seed 0 2>&1 >> logs/catC_fine_"$exp_id".log
)&


(
gpu=3
nl=7
exp_id=""$gpu"_gpu_"$nl""
python train_search.py --max_epochs 20 --hidden_size 128 --batch_size 8192 --catalog_name catB --gpu $gpu --num_layers $nl  --model_path catB_read --exp_id $exp_id --save_searched_file "searched_arch_"$exp_id""   --seed 0 2>&1 >> logs/"catB_search_"$exp_id".log";
python train_sub_network.py --hidden_size 3072 --batch_size 1024 --learning_rate 0.0001 --learning_rate_min 0.0001 --weight_decay 0.0001 --num_layers $nl  --max_epochs 20 --skip_training False --exp_id $exp_id --save_searched_file "searched_arch_"$exp_id""  --catalog_name catB  --model_path catB --gpu $gpu --seed 0 2>&1 >> logs/catB_fine_"$exp_id".log
)&
