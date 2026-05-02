====================================================================================================
saved_time: 2026-05-01 15:56:55
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p002353186_adam0p0004536829_decay104_trial008 --optimizer aux-sign-auto-cos-inc --lrate 0.000453682908579571 --lrate_decay 104 --muon_lrate 0.0023531859828563972 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p002353186_adam0p0004536829_decay104_trial008/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p002353186_adam0p0004536829_decay104_trial008
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 30 min
current_train_loss: 0.0024834820
current_train_psnr: 31.739208
testset_mean_loss: 0.0011946974
testset_mean_psnr: 29.565138
testset_mean_ssim: 0.877095
testset_mean_lpips: 0.085459
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009825138, psnr=30.076613, ssim=0.831170, lpips=0.126233
  image_001: loss=0.0011396535, psnr=29.432271, ssim=0.835212, lpips=0.129407
  image_002: loss=0.0009365905, psnr=30.284502, ssim=0.858215, lpips=0.110643
  image_003: loss=0.0013393137, psnr=28.731177, ssim=0.811657, lpips=0.136650
  image_004: loss=0.0010636862, psnr=29.731864, ssim=0.836176, lpips=0.101172
  image_005: loss=0.0008492812, psnr=30.709484, ssim=0.876219, lpips=0.082684
  image_006: loss=0.0012735873, psnr=28.949712, ssim=0.868482, lpips=0.084921
  image_007: loss=0.0010903314, psnr=29.624415, ssim=0.883484, lpips=0.074203
  image_008: loss=0.0008999927, psnr=30.457610, ssim=0.897443, lpips=0.066074
  image_009: loss=0.0008771720, psnr=30.569152, ssim=0.906468, lpips=0.055500
  image_010: loss=0.0008245275, psnr=30.837948, ssim=0.918615, lpips=0.045584
  image_011: loss=0.0018560562, psnr=27.314089, ssim=0.889589, lpips=0.070807
  image_012: loss=0.0033385023, psnr=24.764483, ssim=0.868628, lpips=0.096551
  image_013: loss=0.0023813001, psnr=26.231859, ssim=0.891426, lpips=0.076314
  image_014: loss=0.0014673329, psnr=28.334713, ssim=0.927009, lpips=0.050676
  image_015: loss=0.0009531568, psnr=30.208356, ssim=0.943745, lpips=0.039831
  image_016: loss=0.0009024104, psnr=30.445959, ssim=0.924334, lpips=0.045791
  image_017: loss=0.0010155449, psnr=29.933008, ssim=0.905016, lpips=0.057119
  image_018: loss=0.0014710977, psnr=28.323585, ssim=0.873575, lpips=0.082366
  image_019: loss=0.0011700500, psnr=29.317955, ssim=0.873440, lpips=0.083474
  image_020: loss=0.0009012959, psnr=30.451325, ssim=0.885414, lpips=0.083624
  image_021: loss=0.0008320110, psnr=30.798709, ssim=0.869071, lpips=0.095218
  image_022: loss=0.0006791580, psnr=31.680291, ssim=0.870209, lpips=0.098682
  image_023: loss=0.0006863498, psnr=31.634545, ssim=0.855336, lpips=0.116632
  image_024: loss=0.0009365190, psnr=30.284834, ssim=0.827455, lpips=0.126324
