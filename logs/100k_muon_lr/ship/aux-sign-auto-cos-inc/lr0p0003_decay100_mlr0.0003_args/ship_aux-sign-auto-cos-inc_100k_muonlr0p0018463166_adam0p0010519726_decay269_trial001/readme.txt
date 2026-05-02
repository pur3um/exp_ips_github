====================================================================================================
saved_time: 2026-04-30 15:06:34
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial001 --optimizer aux-sign-auto-cos-inc --lrate 0.0010519726230559959 --lrate_decay 269 --muon_lrate 0.001846316577722323 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial001/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial001
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 34 min
current_train_loss: 0.0024325654
current_train_psnr: 32.375458
testset_mean_loss: 0.0011457889
testset_mean_psnr: 29.656281
testset_mean_ssim: 0.876357
testset_mean_lpips: 0.085225
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009717769, psnr=30.124334, ssim=0.830479, lpips=0.128168
  image_001: loss=0.0011594237, psnr=29.357578, ssim=0.832734, lpips=0.128015
  image_002: loss=0.0009308493, psnr=30.311206, ssim=0.858574, lpips=0.111657
  image_003: loss=0.0013427188, psnr=28.720149, ssim=0.809417, lpips=0.140846
  image_004: loss=0.0010654394, psnr=29.724712, ssim=0.834510, lpips=0.099376
  image_005: loss=0.0008527803, psnr=30.691628, ssim=0.874726, lpips=0.082421
  image_006: loss=0.0012639593, psnr=28.982669, ssim=0.869033, lpips=0.087836
  image_007: loss=0.0010592914, psnr=29.749845, ssim=0.881641, lpips=0.076261
  image_008: loss=0.0008933953, psnr=30.489563, ssim=0.896801, lpips=0.066921
  image_009: loss=0.0008877841, psnr=30.516926, ssim=0.905643, lpips=0.057335
  image_010: loss=0.0008194040, psnr=30.865019, ssim=0.919127, lpips=0.045400
  image_011: loss=0.0019323792, psnr=27.139076, ssim=0.883794, lpips=0.073692
  image_012: loss=0.0025034146, psnr=26.014672, ssim=0.870659, lpips=0.091621
  image_013: loss=0.0020497744, psnr=26.882939, ssim=0.888832, lpips=0.071743
  image_014: loss=0.0014591863, psnr=28.358892, ssim=0.925708, lpips=0.048648
  image_015: loss=0.0009289896, psnr=30.319891, ssim=0.943509, lpips=0.039742
  image_016: loss=0.0008848996, psnr=30.531059, ssim=0.925474, lpips=0.044425
  image_017: loss=0.0010001906, psnr=29.999172, ssim=0.904883, lpips=0.053373
  image_018: loss=0.0014561758, psnr=28.367862, ssim=0.874893, lpips=0.083629
  image_019: loss=0.0011547754, psnr=29.375024, ssim=0.874185, lpips=0.085805
  image_020: loss=0.0008967136, psnr=30.473462, ssim=0.884252, lpips=0.084920
  image_021: loss=0.0008282738, psnr=30.818260, ssim=0.868017, lpips=0.093666
  image_022: loss=0.0006841781, psnr=31.648308, ssim=0.869963, lpips=0.095481
  image_023: loss=0.0006821266, psnr=31.661349, ssim=0.856371, lpips=0.113846
  image_024: loss=0.0009368225, psnr=30.283426, ssim=0.825689, lpips=0.125796
