====================================================================================================
saved_time: 2026-04-25 22:32:19
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_nerf_ranksched_lazyq.py
executed_command: run_nerf_ranksched_lazyq.py --basedir logs/warmup_cosine_lazyq --expname aux-sign-ranksched-lazy-q_gap1000 --config configs/lego.txt --optimizer aux-sign-ranksched-lazy-q --train_scheduler warmup_cosine --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --lazy_q_update_gap 1000 --lazy_q_rank_refresh_stride 8 --lazy_q_use_b_ema true --lazy_q_b_ema_decay 0.9 --sched_warmup_frac 0.01 --N_iters 200000
expname: aux-sign-ranksched-lazy-q_gap1000
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 6 hour 37 min
current_train_loss: 0.0028656244
current_train_psnr: 31.948166
testset_mean_loss: 0.0007940366
testset_mean_psnr: 31.266943
testset_mean_ssim: 0.961256
testset_mean_lpips: 0.023963
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0006063959, psnr=32.172437, ssim=0.966083, lpips=0.022523
  image_001: loss=0.0016821229, psnr=27.741422, ssim=0.953154, lpips=0.030570
  image_002: loss=0.0004470145, psnr=33.496783, ssim=0.970379, lpips=0.019466
  image_003: loss=0.0005472011, psnr=32.618530, ssim=0.968264, lpips=0.022467
  image_004: loss=0.0008444623, psnr=30.734197, ssim=0.960993, lpips=0.025375
  image_005: loss=0.0005751047, psnr=32.402530, ssim=0.962532, lpips=0.022532
  image_006: loss=0.0008227951, psnr=30.847082, ssim=0.957563, lpips=0.026633
  image_007: loss=0.0005170829, psnr=32.864397, ssim=0.957468, lpips=0.020900
  image_008: loss=0.0005642109, psnr=32.485584, ssim=0.967983, lpips=0.017938
  image_009: loss=0.0011399087, psnr=29.431299, ssim=0.963874, lpips=0.028104
  image_010: loss=0.0016042121, psnr=27.947382, ssim=0.949142, lpips=0.039796
  image_011: loss=0.0011882349, psnr=29.250977, ssim=0.952236, lpips=0.029150
  image_012: loss=0.0007971863, psnr=30.984401, ssim=0.962810, lpips=0.023419
  image_013: loss=0.0008003147, psnr=30.967391, ssim=0.961495, lpips=0.024213
  image_014: loss=0.0006007751, psnr=32.212880, ssim=0.963508, lpips=0.019593
  image_015: loss=0.0007249345, psnr=31.397012, ssim=0.963386, lpips=0.024604
  image_016: loss=0.0007716354, psnr=31.125878, ssim=0.972860, lpips=0.021689
  image_017: loss=0.0007514830, psnr=31.240808, ssim=0.963996, lpips=0.024796
  image_018: loss=0.0007249904, psnr=31.396677, ssim=0.959808, lpips=0.023487
  image_019: loss=0.0006455319, psnr=31.900822, ssim=0.952206, lpips=0.026399
  image_020: loss=0.0004303339, psnr=33.661943, ssim=0.962245, lpips=0.017651
  image_021: loss=0.0006071561, psnr=32.166995, ssim=0.963264, lpips=0.017782
  image_022: loss=0.0008438050, psnr=30.737579, ssim=0.959105, lpips=0.025550
  image_023: loss=0.0008695573, psnr=30.607017, ssim=0.957823, lpips=0.024202
  image_024: loss=0.0007444646, psnr=31.281559, ssim=0.959237, lpips=0.020241
