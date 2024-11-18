output_dir=''
test_dir=''
pretrained_model_name_or_path=' '
traindata_dir=''
test_save_dir=''
test_save_name=''
train_batch_size=4
gradient_accumulation_steps=16
num_train_epochs=90
checkpointing_steps=1000
evaluation_steps=50
learning_rate=3e-5
lr_warmup_steps=0
dataloader_num_workers=8
tracker_project_name='pretrain_tracker'
seed=1234
 
accelerate launch --config_file ../node_config/8gpu.yaml \
                ./train.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --output_dir $output_dir \
                  --test_dir $test_dir \
                  --traindata_dir $traindata_dir \
                  --test_save_dir $test_save_dir \
                  --test_save_name $test_save_name \
                  --checkpointing_steps $checkpointing_steps \
                  --evaluation_steps $evaluation_steps \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --seed $seed \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --enable_xformers_memory_efficient_attention \
                  --use_ema 