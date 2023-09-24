pip install lmdb opencv-python kornia yacs blobfile chumpy face_alignment==1.2.0
pip install Pillow==9.5.0

https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
/root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar
/root/.cache/torch/hub/checkpoints/2DFAN4-11f355bf06.pth.tar


python scripts/align.py -i data/train_image -o data/train_image_256 -s 256
python scripts/create_data.py --data_dir data/train_image --output_dir ffhq256_deca.lmdb --image_size 256 --use_meanshape False --batch_size 32
python scripts/align.py -i PATH_TO_PERSONAL_PHOTO_ALBUM -o personal_images_aligned -s 256
python scripts/create_data.py --data_dir personal_images_aligned --output_dir personal_deca.lmdb --image_size 256 --use_meanshape True

mpiexec -n 8 python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
--log_dir log/stage1 --data_dir ffhq256_deca.lmdb --lr 1e-4 \
--p2_weight True --image_size 256 --batch_size 32 --max_steps 50000 \
--num_workers 8 --save_interval 5000 --stage 1

mpiexec -n 1 python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
--log_dir log/stage2 --resume_checkpoint data/stage1_model050000.pt \
--data_dir peronsal_deca.lmdb --lr 1e-5 \
--p2_weight True --image_size 256 --batch_size 4 --max_steps 5000 \
--num_workers 8 --save_interval 100 --stage 2

python scripts/inference.py --source SOURCE_IMAGE_FILE --target TARGET_IMAGE_FILE \
--output_dir OUTPUT_DIR --modes light --model_path PATH_TO_MODEL \
--meanshape PATH_TO_MEANSHAPE --timestep_respacing ddim20

mpiexec -n 1 python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
--log_dir log/stage1 --data_dir ffhq256_deca.lmdb --lr 1e-4 \
--p2_weight True --image_size 256 --batch_size 8 --max_steps 50000 \
--num_workers 16 --save_interval 100 --stage 1

data_dir="paddle_obama/"
data_dir="torch_obama/"
mkdir -p data/${data_dir}/{aligned,output}
# 原图转256x256大小
python scripts/align.py -i data/pic_obama -o data/${data_dir}/aligned -s 256
# 创建lmdb数据
export FLAGS_set_to_1d=False
python scripts/create_data.py --data_dir data/${data_dir}/aligned --output_dir data/${data_dir}/deca.lmdb \
--image_size 256 --use_meanshape True --batch_size=8
# 训练stage2
mpiexec -n 1 python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
--log_dir log/${data_dir}/stage2 --resume_checkpoint data/${data_dir}/stage1_model050000.pt \
--data_dir data/${data_dir}/deca.lmdb --lr 1e-5 \
--p2_weight True --image_size 256 --batch_size 8 --max_steps 5000 \
--num_workers 8 --save_interval 100 --stage 2
# 推理一张目标图片
python scripts/inference.py --source data/source/obama.png --target data/target/10.png \
--output_dir data/${data_dir}/output --modes pose --model_path log/${data_dir}/stage2/model000100.pt \
--meanshape data/${data_dir}/deca.lmdb/mean_shape.pkl --timestep_respacing ddim20
# 推理多张目标图片
j=0
for i in `ls data/target/*.png`; do
    t="data/${data_dir}/output/${j}"
    mkdir -p $t
    python scripts/inference.py --source data/source/obama.png --target $i \
    --output_dir $t --modes pose,light,exp --model_path log/${data_dir}/stage2/model000500.pt \
    --meanshape data/${data_dir}/deca.lmdb/mean_shape.pkl --timestep_respacing ddim20
    j=$((j+1))
done
