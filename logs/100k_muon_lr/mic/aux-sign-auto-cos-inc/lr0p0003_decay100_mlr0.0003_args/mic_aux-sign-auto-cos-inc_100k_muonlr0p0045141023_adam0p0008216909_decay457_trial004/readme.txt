====================================================================================================
saved_time: 2026-04-30 15:29:55
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial004 --optimizer aux-sign-auto-cos-inc --lrate 0.0008216908785704095 --lrate_decay 457 --muon_lrate 0.004514102265895567 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial004/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial004
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 30 min
current_train_loss: 0.0016834512
current_train_psnr: 33.584312
testset_mean_loss: 0.0004763324
testset_mean_psnr: 33.380908
testset_mean_ssim: 0.979380
testset_mean_lpips: 0.023001
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004541378, psnr=33.428123, ssim=0.979158, lpips=0.013885
  image_001: loss=0.0003027109, psnr=35.189718, ssim=0.983980, lpips=0.020166
  image_002: loss=0.0005018442, psnr=32.994310, ssim=0.977767, lpips=0.026177
  image_003: loss=0.0004359639, psnr=33.605494, ssim=0.977004, lpips=0.027113
  image_004: loss=0.0004477642, psnr=33.489505, ssim=0.974144, lpips=0.028925
  image_005: loss=0.0003854824, psnr=34.139953, ssim=0.978616, lpips=0.028160
  image_006: loss=0.0003441492, psnr=34.632531, ssim=0.990448, lpips=0.009458
  image_007: loss=0.0004367313, psnr=33.597856, ssim=0.983011, lpips=0.011660
  image_008: loss=0.0005496746, psnr=32.598942, ssim=0.970479, lpips=0.038158
  image_009: loss=0.0004493395, psnr=33.474254, ssim=0.974993, lpips=0.023854
  image_010: loss=0.0003170709, psnr=34.988435, ssim=0.982039, lpips=0.016852
  image_011: loss=0.0002892587, psnr=35.387135, ssim=0.985110, lpips=0.018711
  image_012: loss=0.0009900371, psnr=30.043485, ssim=0.979136, lpips=0.020564
  image_013: loss=0.0006449562, psnr=31.904697, ssim=0.983234, lpips=0.015674
  image_014: loss=0.0005473419, psnr=32.617412, ssim=0.982267, lpips=0.016201
  image_015: loss=0.0004633818, psnr=33.340610, ssim=0.980838, lpips=0.019159
  image_016: loss=0.0004171710, psnr=33.796858, ssim=0.978227, lpips=0.020442
  image_017: loss=0.0004793562, psnr=33.193416, ssim=0.975911, lpips=0.028001
  image_018: loss=0.0003377684, psnr=34.713808, ssim=0.984567, lpips=0.012251
  image_019: loss=0.0005218860, psnr=32.824242, ssim=0.988687, lpips=0.009277
  image_020: loss=0.0005617838, psnr=32.504307, ssim=0.973457, lpips=0.038128
  image_021: loss=0.0005525408, psnr=32.576356, ssim=0.971610, lpips=0.042273
  image_022: loss=0.0005565171, psnr=32.545214, ssim=0.972911, lpips=0.035098
  image_023: loss=0.0005596051, psnr=32.521183, ssim=0.975052, lpips=0.032537
  image_024: loss=0.0003618371, psnr=34.414868, ssim=0.981843, lpips=0.022303
