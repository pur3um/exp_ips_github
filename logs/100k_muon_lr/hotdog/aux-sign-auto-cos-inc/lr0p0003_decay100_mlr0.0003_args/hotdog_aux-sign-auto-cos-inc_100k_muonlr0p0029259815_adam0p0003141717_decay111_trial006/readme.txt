====================================================================================================
saved_time: 2026-05-01 14:02:51
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0029259815_adam0p0003141717_decay111_trial006 --optimizer aux-sign-auto-cos-inc --lrate 0.00031417172594439365 --lrate_decay 111 --muon_lrate 0.002925981471069624 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0029259815_adam0p0003141717_decay111_trial006/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0029259815_adam0p0003141717_decay111_trial006
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 28 min
current_train_loss: 0.0009421613
current_train_psnr: 35.952740
testset_mean_loss: 0.0002193388
testset_mean_psnr: 37.379143
testset_mean_ssim: 0.981904
testset_mean_lpips: 0.014271
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001870715, psnr=37.279922, ssim=0.982584, lpips=0.014392
  image_001: loss=0.0001701294, psnr=37.692204, ssim=0.984311, lpips=0.014972
  image_002: loss=0.0002142217, psnr=36.691364, ssim=0.980006, lpips=0.018072
  image_003: loss=0.0001429224, psnr=38.448993, ssim=0.983402, lpips=0.016113
  image_004: loss=0.0001108727, psnr=39.551748, ssim=0.985396, lpips=0.012479
  image_005: loss=0.0001558961, psnr=38.071644, ssim=0.984152, lpips=0.010231
  image_006: loss=0.0002002105, psnr=36.985130, ssim=0.983759, lpips=0.011112
  image_007: loss=0.0001297559, psnr=38.868725, ssim=0.986346, lpips=0.010472
  image_008: loss=0.0001387886, psnr=38.576460, ssim=0.981865, lpips=0.010141
  image_009: loss=0.0001558288, psnr=38.073519, ssim=0.977682, lpips=0.016339
  image_010: loss=0.0001370640, psnr=38.630763, ssim=0.979149, lpips=0.016801
  image_011: loss=0.0002208740, psnr=36.558552, ssim=0.977281, lpips=0.018275
  image_012: loss=0.0002285496, psnr=36.410194, ssim=0.984916, lpips=0.010071
  image_013: loss=0.0001573722, psnr=38.030716, ssim=0.988980, lpips=0.007750
  image_014: loss=0.0002773564, psnr=35.569617, ssim=0.983204, lpips=0.015336
  image_015: loss=0.0011848676, psnr=29.263301, ssim=0.964069, lpips=0.031557
  image_016: loss=0.0004525708, psnr=33.443134, ssim=0.970393, lpips=0.026304
  image_017: loss=0.0001451096, psnr=38.383036, ssim=0.983518, lpips=0.014552
  image_018: loss=0.0002220341, psnr=36.535801, ssim=0.981355, lpips=0.010558
  image_019: loss=0.0001730177, psnr=37.619092, ssim=0.984781, lpips=0.009374
  image_020: loss=0.0001125249, psnr=39.487510, ssim=0.987350, lpips=0.008625
  image_021: loss=0.0001136992, psnr=39.442423, ssim=0.984564, lpips=0.009774
  image_022: loss=0.0001244150, psnr=39.051270, ssim=0.983042, lpips=0.015098
  image_023: loss=0.0001371958, psnr=38.626590, ssim=0.983249, lpips=0.014928
  image_024: loss=0.0001911228, psnr=37.186874, ssim=0.982257, lpips=0.013438
