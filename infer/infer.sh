output_dir=''
test_dir=''
pretrained_model_name_or_path=' '
test_save_dir=''
test_save_name=''
tracker_project_name='pretrain_tracker'
seed=1234
 
accelerate launch --config_file ../node_config/1gpu.yaml \
                ./infer.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --output_dir $output_dir \
                  --test_dir $test_dir \
                  --test_save_dir $test_save_dir \
                  --test_save_name $test_save_name \
                  --seed $seed \
                  --tracker_project_name $tracker_project_name \
                  --enable_xformers_memory_efficient_attention \
                  --use_ema 