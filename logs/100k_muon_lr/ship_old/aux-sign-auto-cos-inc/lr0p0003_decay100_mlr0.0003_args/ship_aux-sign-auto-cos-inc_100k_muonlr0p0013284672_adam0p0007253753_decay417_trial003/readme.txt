====================================================================================================
saved_time: 2026-04-30 22:22:11
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial003 --optimizer aux-sign-auto-cos-inc --lrate 0.0007253753172998836 --lrate_decay 417 --muon_lrate 0.001328467245876384 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial003/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial003
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 31 min
current_train_loss: 0.0025526865
current_train_psnr: 31.984682
testset_mean_loss: 0.0011966544
testset_mean_psnr: 29.484888
testset_mean_ssim: 0.873801
testset_mean_lpips: 0.090879
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0010007524, psnr=29.996733, ssim=0.826626, lpips=0.136479
  image_001: loss=0.0011864317, psnr=29.257572, ssim=0.829164, lpips=0.138076
  image_002: loss=0.0009502132, psnr=30.221789, ssim=0.854981, lpips=0.117131
  image_003: loss=0.0013688874, psnr=28.636323, ssim=0.806485, lpips=0.146098
  image_004: loss=0.0010802245, psnr=29.664859, ssim=0.832840, lpips=0.104521
  image_005: loss=0.0008888114, psnr=30.511904, ssim=0.872044, lpips=0.087736
  image_006: loss=0.0012992008, psnr=28.863237, ssim=0.865729, lpips=0.093590
  image_007: loss=0.0011127243, psnr=29.536124, ssim=0.878488, lpips=0.082465
  image_008: loss=0.0009288308, psnr=30.320634, ssim=0.895150, lpips=0.072350
  image_009: loss=0.0009273031, psnr=30.327782, ssim=0.902436, lpips=0.059621
  image_010: loss=0.0008558786, psnr=30.675878, ssim=0.914463, lpips=0.052735
  image_011: loss=0.0018793360, psnr=27.259956, ssim=0.886075, lpips=0.075725
  image_012: loss=0.0029335232, psnr=25.326105, ssim=0.872799, lpips=0.098125
  image_013: loss=0.0020804349, psnr=26.818458, ssim=0.888038, lpips=0.080161
  image_014: loss=0.0015253646, psnr=28.166263, ssim=0.923257, lpips=0.055239
  image_015: loss=0.0009811831, psnr=30.082499, ssim=0.941207, lpips=0.043148
  image_016: loss=0.0009322296, psnr=30.304771, ssim=0.922774, lpips=0.047806
  image_017: loss=0.0010708763, psnr=29.702607, ssim=0.901815, lpips=0.058899
  image_018: loss=0.0015390582, psnr=28.127449, ssim=0.870349, lpips=0.089582
  image_019: loss=0.0011997595, psnr=29.209058, ssim=0.869844, lpips=0.091335
  image_020: loss=0.0009405357, psnr=30.266247, ssim=0.881131, lpips=0.088135
  image_021: loss=0.0008565930, psnr=30.672254, ssim=0.865107, lpips=0.095720
  image_022: loss=0.0007053515, psnr=31.515944, ssim=0.865166, lpips=0.101996
  image_023: loss=0.0007063447, psnr=31.509832, ssim=0.852574, lpips=0.123756
  image_024: loss=0.0009665116, psnr=30.147929, ssim=0.826475, lpips=0.131543
