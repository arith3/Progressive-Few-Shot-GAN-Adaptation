
echo 
echo "=========================="
echo "Start metric: " 
n_device=0
n_sample=10000
out="./metric_results"
seed=2021

# if you dont have a large target images, eval by this block. (w/o FID)
CUDA_VISIBLE_DEVICES=$n_device python evals.py --out $out --batch 64 \
   --source_ckpt ./source_ffhq.pt --source_key ffhq \
   --model_ckpt ./checkpoints/animation_faces/005000.pt \
   --train_imgs ./processed_data/animation_faces \
   --n_sample $n_sample --seed $seed

# if you have a large target images, eval by this block.  (w IS, IC-LPIPS, FID)
# CUDA_VISIBLE_DEVICES=$n_device python evals.py --out $out --batch 64 \
#    --source_ckpt ./source_ffhq.pt --source_key ffhq \
#    --model_ckpt ./checkpoints/animation_faces/005000.pt \
#    --train_imgs ./processed_data/animation_faces --test_imgs ./raw_data/animation_faces_real/images/ \
#    --n_sample $n_sample --seed $seed