
echo 
echo "=========================="
echo "Start metric: " 
n_device=0
n_sample=100
out="./metric_results"
seed=2021

CUDA_VISIBLE_DEVICES=$n_device python recon_evals.py --out $out --batch 4 \
   --source_ckpt ./source_ffhq.pt --source_key ffhq \
   --model_ckpt ./checkpoints/animation_faces_to_ffhq/005000.pt \
   --n_sample $n_sample --seed $seed
