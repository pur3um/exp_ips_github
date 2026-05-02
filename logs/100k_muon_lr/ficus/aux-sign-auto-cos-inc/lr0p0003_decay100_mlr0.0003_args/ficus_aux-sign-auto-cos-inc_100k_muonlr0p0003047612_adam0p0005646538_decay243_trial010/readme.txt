====================================================================================================
saved_time: 2026-05-01 22:15:19
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0003047612_adam0p0005646538_decay243_trial010 --optimizer aux-sign-auto-cos-inc --lrate 0.0005646537647516755 --lrate_decay 243 --muon_lrate 0.0003047612401310737 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0003047612_adam0p0005646538_decay243_trial010/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0003047612_adam0p0005646538_decay243_trial010
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 28 min
current_train_loss: 0.0035344837
current_train_psnr: 30.712080
testset_mean_loss: 0.0019393011
testset_mean_psnr: 27.242395
testset_mean_ssim: 0.947626
testset_mean_lpips: 0.043909
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0013361603, psnr=28.741414, ssim=0.954608, lpips=0.038076
  image_001: loss=0.0020356462, psnr=26.912977, ssim=0.945034, lpips=0.048091
  image_002: loss=0.0026034797, psnr=25.844458, ssim=0.937672, lpips=0.051743
  image_003: loss=0.0027687182, psnr=25.577212, ssim=0.934003, lpips=0.052580
  image_004: loss=0.0028165937, psnr=25.502758, ssim=0.937074, lpips=0.055739
  image_005: loss=0.0021433856, psnr=26.688997, ssim=0.946984, lpips=0.044225
  image_006: loss=0.0018527467, psnr=27.321839, ssim=0.953372, lpips=0.040470
  image_007: loss=0.0016088613, psnr=27.934814, ssim=0.953836, lpips=0.036206
  image_008: loss=0.0019941940, psnr=27.002326, ssim=0.940931, lpips=0.051490
  image_009: loss=0.0020892103, psnr=26.800178, ssim=0.938618, lpips=0.047482
  image_010: loss=0.0014797184, psnr=28.298209, ssim=0.953831, lpips=0.037672
  image_011: loss=0.0013454467, psnr=28.711335, ssim=0.960376, lpips=0.032925
  image_012: loss=0.0013313937, psnr=28.756935, ssim=0.960304, lpips=0.031665
  image_013: loss=0.0013339865, psnr=28.748485, ssim=0.959549, lpips=0.033982
  image_014: loss=0.0017155962, psnr=27.655849, ssim=0.951291, lpips=0.040336
  image_015: loss=0.0016238140, psnr=27.894637, ssim=0.946816, lpips=0.044691
  image_016: loss=0.0017705425, psnr=27.518936, ssim=0.955838, lpips=0.034918
  image_017: loss=0.0017543263, psnr=27.558896, ssim=0.955312, lpips=0.042731
  image_018: loss=0.0017574906, psnr=27.551070, ssim=0.955633, lpips=0.036516
  image_019: loss=0.0019772945, psnr=27.039286, ssim=0.949616, lpips=0.038162
  image_020: loss=0.0023900466, psnr=26.215936, ssim=0.939837, lpips=0.051501
  image_021: loss=0.0016731893, psnr=27.764549, ssim=0.943902, lpips=0.051017
  image_022: loss=0.0021139532, psnr=26.749046, ssim=0.936428, lpips=0.051754
  image_023: loss=0.0029706843, psnr=25.271435, ssim=0.936586, lpips=0.060139
  image_024: loss=0.0019960487, psnr=26.998288, ssim=0.943193, lpips=0.043622
