====================================================================================================
saved_time: 2026-05-01 00:01:09
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0027592053_adam0p0008216909_decay457_trial002 --optimizer aux-sign-auto-cos-inc --lrate 0.0008216908785704095 --lrate_decay 457 --muon_lrate 0.0027592052977704327 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0027592053_adam0p0008216909_decay457_trial002/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0027592053_adam0p0008216909_decay457_trial002
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0008984003
current_train_psnr: 36.400055
testset_mean_loss: 0.0002329294
testset_mean_psnr: 37.348397
testset_mean_ssim: 0.981930
testset_mean_lpips: 0.014128
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001893206, psnr=37.228019, ssim=0.982543, lpips=0.014215
  image_001: loss=0.0001674829, psnr=37.760292, ssim=0.984570, lpips=0.015278
  image_002: loss=0.0002176225, psnr=36.622959, ssim=0.979754, lpips=0.017914
  image_003: loss=0.0001443288, psnr=38.406468, ssim=0.983000, lpips=0.015653
  image_004: loss=0.0001069531, psnr=39.708062, ssim=0.985744, lpips=0.012860
  image_005: loss=0.0001500514, psnr=38.237596, ssim=0.984584, lpips=0.010411
  image_006: loss=0.0001971299, psnr=37.052474, ssim=0.983958, lpips=0.010669
  image_007: loss=0.0001284506, psnr=38.912635, ssim=0.986537, lpips=0.009731
  image_008: loss=0.0001412355, psnr=38.500558, ssim=0.982001, lpips=0.010535
  image_009: loss=0.0001546542, psnr=38.106379, ssim=0.977679, lpips=0.015954
  image_010: loss=0.0001358088, psnr=38.670717, ssim=0.979346, lpips=0.016062
  image_011: loss=0.0002237193, psnr=36.502963, ssim=0.977127, lpips=0.018365
  image_012: loss=0.0002460289, psnr=36.090137, ssim=0.984623, lpips=0.010330
  image_013: loss=0.0001406853, psnr=38.517509, ssim=0.989572, lpips=0.007362
  image_014: loss=0.0002710662, psnr=35.669246, ssim=0.983510, lpips=0.014829
  image_015: loss=0.0014564150, psnr=28.367148, ssim=0.962791, lpips=0.029862
  image_016: loss=0.0005471752, psnr=32.618735, ssim=0.968960, lpips=0.027781
  image_017: loss=0.0001410118, psnr=38.507442, ssim=0.983721, lpips=0.014876
  image_018: loss=0.0002193899, psnr=36.587832, ssim=0.981596, lpips=0.010642
  image_019: loss=0.0001757758, psnr=37.550407, ssim=0.984512, lpips=0.009867
  image_020: loss=0.0001125703, psnr=39.485756, ssim=0.987410, lpips=0.008759
  image_021: loss=0.0001123458, psnr=39.494426, ssim=0.984731, lpips=0.009623
  image_022: loss=0.0001220355, psnr=39.135136, ssim=0.983509, lpips=0.014745
  image_023: loss=0.0001351951, psnr=38.690388, ssim=0.983634, lpips=0.014200
  image_024: loss=0.0001867828, psnr=37.286630, ssim=0.982841, lpips=0.012671
