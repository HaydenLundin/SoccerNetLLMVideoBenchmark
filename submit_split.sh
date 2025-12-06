#!/bin/bash
#SBATCH --job-name=SoccerSwarm
#SBATCH --partition=GPU            #
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:TitanRTX:4      # Request all 4 GPUs
#SBATCH --output=swarm_%j.log

module load python/3.12.3
module load ffmpeg
source $HOME/soccer_project/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="YOUR_TOKEN" 

# Update time limit to handle multiple videos (e.g., 12 hours)
# #SBATCH --time=12:00:00

echo "🚀 Starting Batch Processing..."

# Loop from 0 to 39 (40 videos)
for i in {0..39}
do
    echo "----------------------------------------------------------------"
    echo "🎬 PROCESSING VIDEO INDEX: $i"
    echo "----------------------------------------------------------------"

    # 1. Launch Swarm (Pass --video_index $i)
    python -u $HOME/soccer_project/qwen_split_worker.py --gpu_id 0 --num_workers 4 --video_index $i &
    python -u $HOME/soccer_project/qwen_split_worker.py --gpu_id 1 --num_workers 4 --video_index $i &
    python -u $HOME/soccer_project/qwen_split_worker.py --gpu_id 2 --num_workers 4 --video_index $i &
    python -u $HOME/soccer_project/qwen_split_worker.py --gpu_id 3 --num_workers 4 --video_index $i &

    # 2. Wait for this video to finish
    wait

    # 3. Merge Results for this video
    python $HOME/soccer_project/merge_results.py --video_index $i

    # 4. Clean up partial files (Optional, saves inode space)
    rm $HOME/soccer_project/partial_results_vid${i}_gpu*.json

    echo "✅ Video $i Complete."
done

echo "🎉 ALL VIDEOS PROCESSED."




