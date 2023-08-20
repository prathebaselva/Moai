#!/bin/sh
#!/bin/bash
#SBATCH --job-name=moai_step1000_archv4_lr1e3_moai23
#SBATCH --output=moai_step1000_archv4_lr1e3_moai23.out
#SBATCH -e moai_step1000_archv4_lr1e3_moai23.err 
#SBATCH -p gypsum-titanx
#SBATCH --mem=20G

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=3-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_999.2
module load cuda11/11.2.1

python train_multi.py --cfg './src/configs/config_moaiparamdiffusion.yml' 
