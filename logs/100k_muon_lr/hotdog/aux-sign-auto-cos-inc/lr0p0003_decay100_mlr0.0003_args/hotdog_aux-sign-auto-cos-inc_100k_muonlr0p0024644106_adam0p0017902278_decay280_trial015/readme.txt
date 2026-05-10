====================================================================================================
saved_time: 2026-05-02 21:53:24
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0024644106_adam0p0017902278_decay280_trial015 --optimizer aux-sign-auto-cos-inc --lrate 0.0017902277711364593 --lrate_decay 280 --muon_lrate 0.0024644105691429435 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0024644106_adam0p0017902278_decay280_trial015/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0024644106_adam0p0017902278_decay280_trial015
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 34 min
current_train_loss: 0.0009292073
current_train_psnr: 36.254646
testset_mean_loss: 0.0002121895
testset_mean_psnr: 37.487610
testset_mean_ssim: 0.982136
testset_mean_lpips: 0.014203
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001891585, psnr=37.231739, ssim=0.982359, lpips=0.013848
  image_001: loss=0.0001720926, psnr=37.642375, ssim=0.984407, lpips=0.015443
  image_002: loss=0.0002179483, psnr=36.616463, ssim=0.979802, lpips=0.022301
  image_003: loss=0.0001404279, psnr=38.525463, ssim=0.983717, lpips=0.015578
  image_004: loss=0.0001065324, psnr=39.725178, ssim=0.985886, lpips=0.012196
  image_005: loss=0.0001491781, psnr=38.262945, ssim=0.984719, lpips=0.010738
  image_006: loss=0.0001928947, psnr=37.146796, ssim=0.983902, lpips=0.010268
  image_007: loss=0.0001291537, psnr=38.888929, ssim=0.986435, lpips=0.009782
  image_008: loss=0.0001349334, psnr=38.698802, ssim=0.981923, lpips=0.010148
  image_009: loss=0.0001534524, psnr=38.140261, ssim=0.977913, lpips=0.016285
  image_010: loss=0.0001345588, psnr=38.710877, ssim=0.979360, lpips=0.016829
  image_011: loss=0.0002227807, psnr=36.521223, ssim=0.976660, lpips=0.019368
  image_012: loss=0.0002236144, psnr=36.505001, ssim=0.984297, lpips=0.010361
  image_013: loss=0.0001458569, psnr=38.360726, ssim=0.989617, lpips=0.008048
  image_014: loss=0.0002397705, psnr=36.202041, ssim=0.984113, lpips=0.013888
  image_015: loss=0.0011145081, psnr=29.529167, ssim=0.965416, lpips=0.029830
  image_016: loss=0.0004420139, psnr=33.545640, ssim=0.971823, lpips=0.025090
  image_017: loss=0.0001408978, psnr=38.510955, ssim=0.983596, lpips=0.014322
  image_018: loss=0.0002131262, psnr=36.713630, ssim=0.981241, lpips=0.010930
  image_019: loss=0.0001701645, psnr=37.691307, ssim=0.984881, lpips=0.009545
  image_020: loss=0.0001110530, psnr=39.544693, ssim=0.987294, lpips=0.008672
  image_021: loss=0.0001126090, psnr=39.484264, ssim=0.984665, lpips=0.009955
  image_022: loss=0.0001243167, psnr=39.054702, ssim=0.983129, lpips=0.014681
  image_023: loss=0.0001352183, psnr=38.689641, ssim=0.983584, lpips=0.014126
  image_024: loss=0.0001884761, psnr=37.247435, ssim=0.982662, lpips=0.012845
