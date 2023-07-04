mkdir ./processed_data

python prepare_data.py --size 256 --out ./processed_data/animation_faces_5s \
   ./raw_data/animation_faces_5s

CUDA_VISIBLE_DEVICE=0 python train.py --ckpt ./source_ffhq.pt \
   --data_path ./processed_data/animation_faces_5s --exp animation_faces_5s_ours \
   --use_ema --use_rel_ema --k 1.0 --extra_ema \
   --use_flow --winsize_inc 1.0002 --winsize 16 --cutoff 0.6 \
   --use_pred --pred_noiseW 2.0 --use_flow_pred --blur 7 \
   --LFC --m 2 --LFCw 0.5 --extra_step 0
