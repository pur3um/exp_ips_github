====================================================================================================
saved_time: 2026-05-01 19:38:48
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0024331892_adam0p0005646538_decay238_trial012 --optimizer aux-sign-auto-cos-inc --lrate 0.0005646537647516755 --lrate_decay 238 --muon_lrate 0.0024331891771541534 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0024331892_adam0p0005646538_decay238_trial012/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0024331892_adam0p0005646538_decay238_trial012
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 28 min
current_train_loss: 0.0013788504
current_train_psnr: 34.591930
testset_mean_loss: 0.0004524147
testset_mean_psnr: 33.645948
testset_mean_ssim: 0.980256
testset_mean_lpips: 0.021976
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004539894, psnr=33.429542, ssim=0.980074, lpips=0.013123
  image_001: loss=0.0002788071, psnr=35.546960, ssim=0.984685, lpips=0.018263
  image_002: loss=0.0004964853, psnr=33.040935, ssim=0.978364, lpips=0.025132
  image_003: loss=0.0004056900, psnr=33.918056, ssim=0.978432, lpips=0.024202
  image_004: loss=0.0004185900, psnr=33.782110, ssim=0.975106, lpips=0.027458
  image_005: loss=0.0003705528, psnr=34.311498, ssim=0.980147, lpips=0.025618
  image_006: loss=0.0003156005, psnr=35.008622, ssim=0.991327, lpips=0.011424
  image_007: loss=0.0004055477, psnr=33.919579, ssim=0.983541, lpips=0.012938
  image_008: loss=0.0005331693, psnr=32.731348, ssim=0.971353, lpips=0.037690
  image_009: loss=0.0004192510, psnr=33.775258, ssim=0.976297, lpips=0.022620
  image_010: loss=0.0002859013, psnr=35.437838, ssim=0.983162, lpips=0.015184
  image_011: loss=0.0002717562, psnr=35.658204, ssim=0.986011, lpips=0.017047
  image_012: loss=0.0010837705, psnr=29.650626, ssim=0.977000, lpips=0.020278
  image_013: loss=0.0006450150, psnr=31.904301, ssim=0.982819, lpips=0.016457
  image_014: loss=0.0004832995, psnr=33.157835, ssim=0.983331, lpips=0.016625
  image_015: loss=0.0004370571, psnr=33.594617, ssim=0.981867, lpips=0.018724
  image_016: loss=0.0003870470, psnr=34.122362, ssim=0.978714, lpips=0.017932
  image_017: loss=0.0004367979, psnr=33.597194, ssim=0.976662, lpips=0.026376
  image_018: loss=0.0003119284, psnr=35.059450, ssim=0.985455, lpips=0.012681
  image_019: loss=0.0004717631, psnr=33.262760, ssim=0.989445, lpips=0.011790
  image_020: loss=0.0005147682, psnr=32.883882, ssim=0.974090, lpips=0.038625
  image_021: loss=0.0005160311, psnr=32.873241, ssim=0.973201, lpips=0.039298
  image_022: loss=0.0005123706, psnr=32.904157, ssim=0.974942, lpips=0.031264
  image_023: loss=0.0005180547, psnr=32.856243, ssim=0.977202, lpips=0.028665
  image_024: loss=0.0003371248, psnr=34.722091, ssim=0.983176, lpips=0.019991
