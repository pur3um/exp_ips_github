====================================================================================================
saved_time: 2026-05-01 08:53:27
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p0003636926_adam0p0003141717_decay227_trial006 --optimizer aux-sign-auto-cos-inc --lrate 0.00031417172594439365 --lrate_decay 227 --muon_lrate 0.0003636925892547774 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p0003636926_adam0p0003141717_decay227_trial006/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p0003636926_adam0p0003141717_decay227_trial006
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 31 min
current_train_loss: 0.0038972276
current_train_psnr: 29.156990
testset_mean_loss: 0.0017532731
testset_mean_psnr: 27.753472
testset_mean_ssim: 0.837462
testset_mean_lpips: 0.139076
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0014876013, psnr=28.275134, ssim=0.764627, lpips=0.205479
  image_001: loss=0.0017595509, psnr=27.545981, ssim=0.766081, lpips=0.204811
  image_002: loss=0.0015304749, psnr=28.151738, ssim=0.787224, lpips=0.182429
  image_003: loss=0.0017524180, psnr=27.563623, ssim=0.766577, lpips=0.195481
  image_004: loss=0.0015107115, psnr=28.208184, ssim=0.799209, lpips=0.155138
  image_005: loss=0.0014026716, psnr=28.530440, ssim=0.836375, lpips=0.134480
  image_006: loss=0.0018328532, psnr=27.368723, ssim=0.835950, lpips=0.143022
  image_007: loss=0.0017280745, psnr=27.624375, ssim=0.847233, lpips=0.126033
  image_008: loss=0.0015161864, psnr=28.192474, ssim=0.860827, lpips=0.118909
  image_009: loss=0.0015384656, psnr=28.129122, ssim=0.872482, lpips=0.101551
  image_010: loss=0.0013665644, psnr=28.643699, ssim=0.889882, lpips=0.085353
  image_011: loss=0.0021965071, psnr=26.582674, ssim=0.866457, lpips=0.107586
  image_012: loss=0.0039543542, psnr=24.029244, ssim=0.838076, lpips=0.148116
  image_013: loss=0.0029929960, psnr=25.238938, ssim=0.854930, lpips=0.128396
  image_014: loss=0.0022997961, psnr=26.383106, ssim=0.889329, lpips=0.099085
  image_015: loss=0.0015445384, psnr=28.112013, ssim=0.909840, lpips=0.083111
  image_016: loss=0.0013749093, psnr=28.617259, ssim=0.893517, lpips=0.082763
  image_017: loss=0.0016785797, psnr=27.750580, ssim=0.872193, lpips=0.104222
  image_018: loss=0.0021658724, psnr=26.643671, ssim=0.845850, lpips=0.138721
  image_019: loss=0.0017802153, psnr=27.495274, ssim=0.838555, lpips=0.136323
  image_020: loss=0.0015065128, psnr=28.220272, ssim=0.846344, lpips=0.135210
  image_021: loss=0.0013817050, psnr=28.595846, ssim=0.829727, lpips=0.143457
  image_022: loss=0.0011221581, psnr=29.499459, ssim=0.829146, lpips=0.150750
  image_023: loss=0.0010779399, psnr=29.674054, ssim=0.814923, lpips=0.173655
  image_024: loss=0.0013301722, psnr=28.760921, ssim=0.781186, lpips=0.192810
