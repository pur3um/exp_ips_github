====================================================================================================
saved_time: 2026-04-30 11:31:59
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p0016353358_adam0p0010615361_decay386_trial000 --optimizer aux-sign-auto-cos-inc --lrate 0.0010615360772719654 --lrate_decay 386 --muon_lrate 0.001635335754309247 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p0016353358_adam0p0010615361_decay386_trial000/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p0016353358_adam0p0010615361_decay386_trial000
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 37 min
current_train_loss: 0.0026582144
current_train_psnr: 31.226044
testset_mean_loss: 0.0011726938
testset_mean_psnr: 29.584713
testset_mean_ssim: 0.876339
testset_mean_lpips: 0.087470
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009763585, psnr=30.103907, ssim=0.830248, lpips=0.130266
  image_001: loss=0.0011377182, psnr=29.439653, ssim=0.835198, lpips=0.131669
  image_002: loss=0.0009247490, psnr=30.339761, ssim=0.858977, lpips=0.113933
  image_003: loss=0.0013473047, psnr=28.705342, ssim=0.809413, lpips=0.137066
  image_004: loss=0.0010812079, psnr=29.660908, ssim=0.834550, lpips=0.101477
  image_005: loss=0.0008587622, psnr=30.661270, ssim=0.875227, lpips=0.087088
  image_006: loss=0.0013035884, psnr=28.848595, ssim=0.867969, lpips=0.090664
  image_007: loss=0.0011014850, psnr=29.580214, ssim=0.881850, lpips=0.075511
  image_008: loss=0.0008942580, psnr=30.485371, ssim=0.897369, lpips=0.069495
  image_009: loss=0.0009073105, psnr=30.422440, ssim=0.905500, lpips=0.057948
  image_010: loss=0.0008441404, psnr=30.735853, ssim=0.917861, lpips=0.047754
  image_011: loss=0.0017096660, psnr=27.670887, ssim=0.890994, lpips=0.072215
  image_012: loss=0.0029014167, psnr=25.373899, ssim=0.871720, lpips=0.096843
  image_013: loss=0.0021726426, psnr=26.630117, ssim=0.888506, lpips=0.078536
  image_014: loss=0.0015258109, psnr=28.164993, ssim=0.924426, lpips=0.051147
  image_015: loss=0.0009341923, psnr=30.295637, ssim=0.942862, lpips=0.041537
  image_016: loss=0.0008656139, psnr=30.626757, ssim=0.924850, lpips=0.045566
  image_017: loss=0.0010321558, psnr=29.862547, ssim=0.904180, lpips=0.057427
  image_018: loss=0.0015404171, psnr=28.123616, ssim=0.872964, lpips=0.091023
  image_019: loss=0.0011783931, psnr=29.287098, ssim=0.873123, lpips=0.087975
  image_020: loss=0.0009025862, psnr=30.445113, ssim=0.884364, lpips=0.084274
  image_021: loss=0.0008471196, psnr=30.720552, ssim=0.866874, lpips=0.093723
  image_022: loss=0.0006868253, psnr=31.631537, ssim=0.868973, lpips=0.097244
  image_023: loss=0.0006995509, psnr=31.551806, ssim=0.854481, lpips=0.119576
  image_024: loss=0.0009440723, psnr=30.249947, ssim=0.825993, lpips=0.126801
