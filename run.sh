# run base model training
torchrun --nproc_per_node 8 --master_port 12345 main.py --warmup_epochs 2 \
         --model base --epochs 400 --epoch_size 100_000 --batch_size 4 \
         --lr 2e-4 --input_size_list 224 384 512 --log_dir output_dir/base \
         --output_dir output_dir/base \
         --mask_mode random rectangle ellipse \
         --mask_ratio 0.9 0.75 0.75 \
         --dynamic_batch
