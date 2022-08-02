# Hardware
num_worker=2

# CNN BackBone
backbone="resnet50"

# Input Size
input_size=112

# Data Path
# - Verification
verify_base_dir="./dataset/verification"
verify_df_path="./dataset/verification/verification_dev.csv"

# Distance Metrics
distance="cosine"

# Training Hyper-Parameter
batch_size=64

# Trained Model Path
ckpt_load="./checkpoints/ckpt__final"

# Report
report="./reports"

python verification.py \
        --num-worker $num_worker \
        --gpu \
        --backbone $backbone \
        --input-size $input_size \
        --verify-base-dir $verify_base_dir \
        --verify-df-path $verify_df_path \
        --distance $distance \
        --batch-size $batch_size \
        --ckpt-load $ckpt_load \
        --report $report
