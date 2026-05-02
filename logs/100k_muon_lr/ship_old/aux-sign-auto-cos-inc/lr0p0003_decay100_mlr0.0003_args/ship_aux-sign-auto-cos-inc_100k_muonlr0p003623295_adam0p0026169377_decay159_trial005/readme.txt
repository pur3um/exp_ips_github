====================================================================================================
saved_time: 2026-05-01 05:21:49
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p003623295_adam0p0026169377_decay159_trial005 --optimizer aux-sign-auto-cos-inc --lrate 0.002616937682010607 --lrate_decay 159 --muon_lrate 0.003623295039993152 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p003623295_adam0p0026169377_decay159_trial005/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p003623295_adam0p0026169377_decay159_trial005
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0028714896
current_train_psnr: 30.870056
testset_mean_loss: 0.0012600194
testset_mean_psnr: 29.379337
testset_mean_ssim: 0.874424
testset_mean_lpips: 0.088840
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009990799, psnr=30.003997, ssim=0.827363, lpips=0.126208
  image_001: loss=0.0012003819, psnr=29.206805, ssim=0.828457, lpips=0.142916
  image_002: loss=0.0009915145, psnr=30.037009, ssim=0.850507, lpips=0.116428
  image_003: loss=0.0013466432, psnr=28.707474, ssim=0.809979, lpips=0.142361
  image_004: loss=0.0010688592, psnr=29.710794, ssim=0.835163, lpips=0.106890
  image_005: loss=0.0008965662, psnr=30.474176, ssim=0.872122, lpips=0.086533
  image_006: loss=0.0012672871, psnr=28.971250, ssim=0.869809, lpips=0.086104
  image_007: loss=0.0011034042, psnr=29.572653, ssim=0.882891, lpips=0.077073
  image_008: loss=0.0009461627, psnr=30.240341, ssim=0.895082, lpips=0.068774
  image_009: loss=0.0009110871, psnr=30.404400, ssim=0.904610, lpips=0.059439
  image_010: loss=0.0008688970, psnr=30.610316, ssim=0.918768, lpips=0.047858
  image_011: loss=0.0024745048, psnr=26.065117, ssim=0.882991, lpips=0.071640
  image_012: loss=0.0038389950, psnr=24.157824, ssim=0.862529, lpips=0.100216
  image_013: loss=0.0022396464, psnr=26.498205, ssim=0.889245, lpips=0.072236
  image_014: loss=0.0015381083, psnr=28.130130, ssim=0.923961, lpips=0.050284
  image_015: loss=0.0009650158, psnr=30.154655, ssim=0.941551, lpips=0.042887
  image_016: loss=0.0009162338, psnr=30.379937, ssim=0.923391, lpips=0.046069
  image_017: loss=0.0010540551, psnr=29.771367, ssim=0.903482, lpips=0.058109
  image_018: loss=0.0014546008, psnr=28.372561, ssim=0.876424, lpips=0.085229
  image_019: loss=0.0011818667, psnr=29.274315, ssim=0.872922, lpips=0.089056
  image_020: loss=0.0009480008, psnr=30.231912, ssim=0.882269, lpips=0.088486
  image_021: loss=0.0008816324, psnr=30.547124, ssim=0.865162, lpips=0.097601
  image_022: loss=0.0007402374, psnr=31.306289, ssim=0.863800, lpips=0.104457
  image_023: loss=0.0007236233, psnr=31.404874, ssim=0.852012, lpips=0.124099
  image_024: loss=0.0009440816, psnr=30.249904, ssim=0.826108, lpips=0.130056
