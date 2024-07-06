#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --account=p200493
#SBATCH --qos=default
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.anojan@insight-centre.org
#SBATCH --job-name=training_hourglass
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=128
cd $SLURM_SUBMIT_DIR
module load Python/3.11.3-GCCcore-12.3.0
#python -m venv stylegan
source stylegan/bin/activate
#python -m pip install -r requirements.txt

export CUDA_PATH=/apps/USE/easybuild/release/2023.1/software/CUDA/12.1.1
export PATH=$CUDA_PATH/bin:$PATH 
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

srun -n 1 --exact python train.py --outdir=./training_results/ --gpus=4 --batch=64 --metrics=fid2993_full --data=/project/home/p200493/KidsMiccaiDataset/croplip_yflw/imgs/ --dataloader=datasets.dataset_256.ImageFolderMaskDataset --workers=3 --mirror=True --cond=False --cfg=places256 --aug=ada --generator=networks.mat.Generator --discriminator=networks.mat.Discriminator --loss=losses.loss.TwoStageLoss --pr=0.1 --pl=False --truncation=0.5 --style_mix=0.5 --ema=10 --lr=0.001 > stylegan_yflw_randommask.out


#srun -n 1 --exact python train.py --outdir=./training_results/ --gpus=1 --batch=8 --metrics=fid36k5_full --data=/project/home/p200493/KidsMiccaiDataset/croplip_mask/imgs/ --dataloader=datasets.dataset_256.ImageFolderMaskDataset --mirror=True --cond=False --cfg=paper256 --aug=noaug --generator=networks.mat.Generator --discriminator=networks.mat.Discriminator --loss=losses.loss.TwoStageLoss --pr=0.1 --pl=False --truncation=0.5 --style_mix=0.5 --ema=10 --lr=0.001 > stylegan_new.out
