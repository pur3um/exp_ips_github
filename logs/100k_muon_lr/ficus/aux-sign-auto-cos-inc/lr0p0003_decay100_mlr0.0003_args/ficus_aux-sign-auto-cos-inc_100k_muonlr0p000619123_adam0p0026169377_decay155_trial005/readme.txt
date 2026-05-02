====================================================================================================
saved_time: 2026-05-01 04:55:46
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p000619123_adam0p0026169377_decay155_trial005 --optimizer aux-sign-auto-cos-inc --lrate 0.002616937682010607 --lrate_decay 155 --muon_lrate 0.0006191229698148474 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p000619123_adam0p0026169377_decay155_trial005/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p000619123_adam0p0026169377_decay155_trial005
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 28 min
current_train_loss: 0.0026845529
current_train_psnr: 31.980555
testset_mean_loss: 0.0013949314
testset_mean_psnr: 28.696382
testset_mean_ssim: 0.961960
testset_mean_lpips: 0.030288
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009778960, psnr=30.097073, ssim=0.966978, lpips=0.025674
  image_001: loss=0.0014852580, psnr=28.281981, ssim=0.961317, lpips=0.034372
  image_002: loss=0.0017844565, psnr=27.484940, ssim=0.955589, lpips=0.034812
  image_003: loss=0.0020101364, psnr=26.967745, ssim=0.949959, lpips=0.037691
  image_004: loss=0.0020903556, psnr=26.797798, ssim=0.950887, lpips=0.038297
  image_005: loss=0.0014409733, psnr=28.413440, ssim=0.961426, lpips=0.037602
  image_006: loss=0.0013433914, psnr=28.717974, ssim=0.964470, lpips=0.027910
  image_007: loss=0.0010196333, psnr=29.915560, ssim=0.969113, lpips=0.023935
  image_008: loss=0.0015063914, psnr=28.220621, ssim=0.956554, lpips=0.035332
  image_009: loss=0.0017399481, psnr=27.594637, ssim=0.951409, lpips=0.032474
  image_010: loss=0.0010652848, psnr=29.725342, ssim=0.968527, lpips=0.023034
  image_011: loss=0.0009253349, psnr=30.337010, ssim=0.974347, lpips=0.019175
  image_012: loss=0.0009378531, psnr=30.278651, ssim=0.973239, lpips=0.029117
  image_013: loss=0.0008618617, psnr=30.645624, ssim=0.974941, lpips=0.018772
  image_014: loss=0.0012645496, psnr=28.980641, ssim=0.965685, lpips=0.024470
  image_015: loss=0.0012462928, psnr=29.043799, ssim=0.960692, lpips=0.029794
  image_016: loss=0.0012862239, psnr=28.906834, ssim=0.969177, lpips=0.024986
  image_017: loss=0.0012264146, psnr=29.113627, ssim=0.967783, lpips=0.032416
  image_018: loss=0.0013500644, psnr=28.696455, ssim=0.965899, lpips=0.026619
  image_019: loss=0.0014474620, psnr=28.393928, ssim=0.962680, lpips=0.026289
  image_020: loss=0.0016732219, psnr=27.764464, ssim=0.954713, lpips=0.037529
  image_021: loss=0.0009996416, psnr=30.001556, ssim=0.963589, lpips=0.030790
  image_022: loss=0.0015642663, psnr=28.056893, ssim=0.953459, lpips=0.035168
  image_023: loss=0.0021381665, psnr=26.699585, ssim=0.950443, lpips=0.040767
  image_024: loss=0.0014882078, psnr=28.273364, ssim=0.956128, lpips=0.030188
