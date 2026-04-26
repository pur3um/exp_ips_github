====================================================================================================
saved_time: 2026-04-25 19:27:59
script_path: /workspace/exp_ips_github/run_nerf_ranksched_lazyq.py
executed_command: run_nerf_ranksched_lazyq.py --basedir logs/warmup_cosine_lazyq --expname aux-sign-ranksched-lazy-q_gap1000 --config configs/lego.txt --optimizer aux-sign-ranksched-lazy-q --train_scheduler warmup_cosine --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --lazy_q_update_gap 1000 --lazy_q_rank_refresh_stride 8 --lazy_q_use_b_ema true --lazy_q_b_ema_decay 0.9 --sched_warmup_frac 0.01
expname: aux-sign-ranksched-lazy-q_gap1000
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 42 min
current_train_loss: 0.0036547231
current_train_psnr: 30.025211
testset_mean_loss: 0.0010874484
testset_mean_psnr: 29.848941
testset_mean_ssim: 0.946660
testset_mean_lpips: 0.038454
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008512588, psnr=30.699383, ssim=0.952056, lpips=0.036633
  image_001: loss=0.0020874308, psnr=26.803879, ssim=0.940114, lpips=0.040988
  image_002: loss=0.0006055151, psnr=32.178750, ssim=0.959457, lpips=0.030282
  image_003: loss=0.0007608188, psnr=31.187187, ssim=0.955950, lpips=0.038307
  image_004: loss=0.0010761480, psnr=29.681280, ssim=0.948500, lpips=0.042233
  image_005: loss=0.0008451212, psnr=30.730809, ssim=0.944960, lpips=0.038466
  image_006: loss=0.0011346178, psnr=29.451504, ssim=0.939029, lpips=0.041027
  image_007: loss=0.0007964830, psnr=30.988235, ssim=0.937464, lpips=0.038344
  image_008: loss=0.0008297993, psnr=30.810269, ssim=0.953682, lpips=0.031531
  image_009: loss=0.0016085694, psnr=27.935602, ssim=0.949369, lpips=0.045063
  image_010: loss=0.0020578105, psnr=26.865946, ssim=0.934826, lpips=0.054377
  image_011: loss=0.0015569406, psnr=28.077279, ssim=0.939140, lpips=0.043051
  image_012: loss=0.0010655994, psnr=29.724060, ssim=0.951922, lpips=0.034636
  image_013: loss=0.0010801661, psnr=29.665094, ssim=0.949680, lpips=0.035133
  image_014: loss=0.0008186460, psnr=30.869038, ssim=0.950501, lpips=0.031039
  image_015: loss=0.0009678308, psnr=30.142005, ssim=0.951758, lpips=0.037404
  image_016: loss=0.0011597360, psnr=29.356408, ssim=0.960087, lpips=0.036751
  image_017: loss=0.0010268328, psnr=29.885002, ssim=0.951635, lpips=0.038723
  image_018: loss=0.0010515595, psnr=29.781661, ssim=0.941686, lpips=0.038060
  image_019: loss=0.0008410296, psnr=30.751887, ssim=0.934086, lpips=0.039967
  image_020: loss=0.0006758910, psnr=31.701233, ssim=0.942025, lpips=0.036558
  image_021: loss=0.0008608015, psnr=30.650969, ssim=0.947182, lpips=0.033376
  image_022: loss=0.0011836543, psnr=29.267751, ssim=0.944295, lpips=0.043281
  image_023: loss=0.0011940268, psnr=29.229859, ssim=0.942620, lpips=0.041326
  image_024: loss=0.0010499215, psnr=29.788431, ssim=0.944475, lpips=0.034795
