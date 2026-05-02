====================================================================================================
saved_time: 2026-04-30 15:01:29
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial001 --optimizer aux-sign-auto-cos-inc --lrate 0.0010519726230559959 --lrate_decay 269 --muon_lrate 0.001846316577722323 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial001/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial001
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 32 min
current_train_loss: 0.0021213691
current_train_psnr: 33.196480
testset_mean_loss: 0.0012340782
testset_mean_psnr: 29.257225
testset_mean_ssim: 0.966853
testset_mean_lpips: 0.023256
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008295045, psnr=30.811812, ssim=0.971846, lpips=0.020351
  image_001: loss=0.0013822161, psnr=28.594240, ssim=0.965251, lpips=0.024367
  image_002: loss=0.0015186362, psnr=28.185462, ssim=0.961792, lpips=0.027036
  image_003: loss=0.0017624568, psnr=27.538815, ssim=0.956543, lpips=0.030159
  image_004: loss=0.0018378396, psnr=27.356924, ssim=0.958518, lpips=0.030928
  image_005: loss=0.0012478282, psnr=29.038452, ssim=0.966474, lpips=0.025986
  image_006: loss=0.0012951806, psnr=28.876696, ssim=0.966515, lpips=0.024498
  image_007: loss=0.0008334720, psnr=30.791089, ssim=0.974893, lpips=0.016481
  image_008: loss=0.0012774859, psnr=28.936439, ssim=0.962509, lpips=0.025970
  image_009: loss=0.0015717429, psnr=28.036185, ssim=0.956110, lpips=0.028820
  image_010: loss=0.0009326306, psnr=30.302903, ssim=0.973369, lpips=0.017617
  image_011: loss=0.0008214168, psnr=30.854364, ssim=0.978615, lpips=0.014279
  image_012: loss=0.0007970812, psnr=30.984974, ssim=0.978000, lpips=0.014489
  image_013: loss=0.0007186990, psnr=31.434529, ssim=0.979317, lpips=0.015006
  image_014: loss=0.0011357998, psnr=29.446982, ssim=0.969741, lpips=0.021426
  image_015: loss=0.0011177278, psnr=29.516639, ssim=0.965446, lpips=0.023974
  image_016: loss=0.0011471477, psnr=29.403806, ssim=0.973859, lpips=0.016518
  image_017: loss=0.0010636962, psnr=29.731823, ssim=0.972693, lpips=0.021304
  image_018: loss=0.0011116972, psnr=29.540135, ssim=0.969626, lpips=0.020316
  image_019: loss=0.0013079731, psnr=28.834012, ssim=0.966307, lpips=0.021488
  image_020: loss=0.0015305385, psnr=28.151557, ssim=0.961156, lpips=0.030059
  image_021: loss=0.0008093914, psnr=30.918414, ssim=0.970188, lpips=0.022577
  image_022: loss=0.0013564106, psnr=28.676088, ssim=0.958895, lpips=0.028735
  image_023: loss=0.0020813423, psnr=26.816565, ssim=0.953370, lpips=0.033294
  image_024: loss=0.0013640387, psnr=28.651733, ssim=0.960287, lpips=0.025724
