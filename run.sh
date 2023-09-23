python scripts/align.py -i data/train_image -o data/train_image_256 -s 256
python scripts/create_data.py --data_dir data/train_image_256 --output_dir ffhq256_deca.lmdb --image_size 256 --use_meanshape False
python scripts/align.py -i PATH_TO_PERSONAL_PHOTO_ALBUM -o personal_images_aligned -s 256
python scripts/create_data.py --data_dir personal_images_aligned --output_dir personal_deca.lmdb --image_size 256 --use_meanshape True
mpiexec -n 8 python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
    --log_dir log/stage1 --data_dir ffhq256_deca.lmdb --lr 1e-4 \
    --p2_weight True --image_size 256 --batch_size 32 --max_steps 50000 \
    --num_workers 8 --save_interval 5000 --stage 1
mpiexec -n 1 python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
    --log_dir log/stage2 --resume_checkpoint log/stage1/[MODEL_NAME].pt \
    --data_dir peronsal_deca.lmdb --lr 1e-5 \
    --p2_weight True --image_size 256 --batch_size 4 --max_steps 5000 \
    --num_workers 8 --save_interval 5000 --stage 2
python scripts/inference.py --source SOURCE_IMAGE_FILE --target TARGET_IMAGE_FILE --output_dir OUTPUT_DIR --modes light --model_path PATH_TO_MODEL --meanshape PATH_TO_MEANSHAPE --timestep_respacing ddim20
