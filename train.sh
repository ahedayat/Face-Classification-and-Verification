# Hardware
num_worker=3

# CNN BackBone
backbone="resnet_gray"

# Input Size
input_size=112

# Data Path
# - Train
train_base_dir="./dataset/classification/train1000"
train_df_path="./dataset/classification/train.csv"
# - Validation
val_base_dir="./dataset/classification/test1000"
val_df_path="./dataset/classification/test.csv"

# Loss Function
criterion="cross_entropy"

# Optimizer and its hyper-parameters
optimizer="adam"
learning_rate=1e-5

# Training Hyper-Parameter
epoch=100
batch_size=32

# Saving Paths
save_freq=5
ckpt_path="./checkpoints"
ckpt_prefix="ckpt_"
ckpt_load="./checkpoints/ckpt__epoch_49"

# Report
report="./reports"

python classification_train.py \
        --num-worker $num_worker \
        --gpu \
        --backbone $backbone \
        --input-size $input_size \
        --train-base-dir $train_base_dir \
        --train-df-path $train_df_path \
        --val-base-dir $val_base_dir \
        --val-df-path $val_df_path \
        --criterion  $criterion \
        --optimizer $optimizer \
        --learning-rate $learning_rate \
        --epoch $epoch \
        --batch-size $batch_size \
        --save-freq $save_freq \
        --ckpt-path $ckpt_path \
        --ckpt-prefix $ckpt_prefix \
        --report $report
