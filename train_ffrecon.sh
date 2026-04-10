# muskie_large + 40 layers decoder
COMMAND="HYDRA_FULL_ERROR=1 torchrun --nproc_per_node 8 train_ffrecon.py train.epochs=60 enable_checkpoint=True \
        model_name=muskie_large \
        model.decoder_depth=20 \
        train.batch_size=4 \
        train.optimizer.warmup_epochs=2 \
        train.optimizer.lr=4e-5 \
        paths.output_dir=./output_dir/ffrecon/muskie_large_40layers/ \
        train.resume=./output_dir/ffrecon/muskie_large_40layers/checkpoint-latest.pth \
        paths.root_data=/home/pdl/liusidun_3d/lwy_3d/dust3r_datasets
"
# muskie_large + 10 layers decoder
COMMAND="HYDRA_FULL_ERROR=1 torchrun --nproc_per_node 8 train_ffrecon.py train.epochs=60 enable_checkpoint=True \
        model_name=muskie_large \
        model.decoder_depth=10 \
        train.batch_size=6 \
        train.optimizer.warmup_epochs=2 \
        train.optimizer.lr=5e-5 \
        paths.output_dir=./output_dir/ffrecon/muskie_large_20layers/ \
        train.resume=./output_dir/ffrecon/muskie_large_20layers/checkpoint-latest.pth \
        paths.root_data=/home/pdl/liusidun_3d/lwy_3d/dust3r_datasets
"

# dinov3 + 10 layers decoder
COMMAND="HYDRA_FULL_ERROR=1 torchrun --nproc_per_node 8 train_ffrecon.py train.epochs=60 enable_checkpoint=True \
        model_name=dinov3 \
        model.decoder_depth=10 \
        train.batch_size=2 \
        train.optimizer.warmup_epochs=2 \
        train.optimizer.lr=5e-5 \
        paths.output_dir=./output_dir/ffrecon/dinov3_10layers/ \
        train.resume=./output_dir/ffrecon/dinov3_10layers/checkpoint-latest.pth \
        paths.root_data=/home/pdl/liusidun_3d/lwy_3d/dust3r_datasets
"

# dinov3 + 14 layers decoder
COMMAND="HYDRA_FULL_ERROR=1 torchrun --nproc_per_node 8 train_ffrecon.py train.epochs=60 enable_checkpoint=True \
        model_name=dinov3 \
        model.decoder_depth=14 \
        train.batch_size=4 \
        train.optimizer.warmup_epochs=2 \
        train.optimizer.lr=5e-5 \
        paths.output_dir=/data6T1/lwy/output_dir_2/ffrecon/dinov3_28layers/ \
        train.resume=/data6T1/lwy/output_dir_2/ffrecon/dinov3_28layers/checkpoint-latest.pth \
        paths.root_data=/home/pdl/liusidun_3d/lwy_3d/dust3r_datasets
"
# python -m debugpy --listen 5679 /home/pdl/miniconda3/envs/spann3r/bin/
# muskie_large + 14 layers decoder (ffrecon_larger)
COMMAND="HYDRA_FULL_ERROR=1 torchrun --nproc_per_node 8 train_ffrecon.py --config-name ffrecon_larger train.epochs=100 enable_checkpoint=True \
        model_name=muskie_large \
        model.decoder_depth=14 \
        train.batch_size=6 \
        train.optimizer.warmup_epochs=2 \
        train.optimizer.lr=5e-5 \
        paths.output_dir=/data6T1/lwy/output_dir_2/ffrecon/muskie_large_28layers/ \
        train.resume=/data6T1/lwy/output_dir_2/ffrecon/muskie_large_28layers/checkpoint-latest.pth \
        paths.root_data=/home/pdl/liusidun_3d/lwy_3d/dust3r_datasets
"

while true; do
    echo "--- Starting or Retrying the training command at $(date) ---"
    eval $COMMAND
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "--- Command finished successfully with exit code 0. Exiting loop. ---"
        break
    else
        echo "--- Command failed with exit code $EXIT_CODE. Retrying in 5 seconds... ---"
        sleep 2
    fi
done