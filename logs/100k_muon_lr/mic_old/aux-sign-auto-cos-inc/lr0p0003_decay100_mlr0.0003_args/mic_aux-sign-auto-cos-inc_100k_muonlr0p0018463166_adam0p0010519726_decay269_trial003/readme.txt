====================================================================================================
saved_time: 2026-04-30 11:58:39
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial003 --optimizer aux-sign-auto-cos-inc --lrate 0.0010519726230559959 --lrate_decay 269 --muon_lrate 0.001846316577722323 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial003/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial003
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 36 min
current_train_loss: 0.0014226530
current_train_psnr: 34.178432
testset_mean_loss: 0.0004496510
testset_mean_psnr: 33.644298
testset_mean_ssim: 0.980087
testset_mean_lpips: 0.021882
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004411475, psnr=33.554161, ssim=0.979423, lpips=0.014265
  image_001: loss=0.0002755708, psnr=35.597666, ssim=0.984656, lpips=0.017828
  image_002: loss=0.0005292685, psnr=32.763239, ssim=0.976904, lpips=0.023731
  image_003: loss=0.0004170342, psnr=33.798282, ssim=0.977634, lpips=0.025015
  image_004: loss=0.0004148840, psnr=33.820733, ssim=0.975536, lpips=0.026374
  image_005: loss=0.0003686125, psnr=34.334299, ssim=0.980278, lpips=0.024078
  image_006: loss=0.0003125693, psnr=35.050535, ssim=0.991384, lpips=0.011545
  image_007: loss=0.0004164373, psnr=33.804503, ssim=0.983122, lpips=0.013587
  image_008: loss=0.0005221686, psnr=32.821891, ssim=0.971514, lpips=0.036473
  image_009: loss=0.0004322691, psnr=33.642457, ssim=0.976323, lpips=0.021776
  image_010: loss=0.0002852318, psnr=35.448019, ssim=0.982923, lpips=0.015094
  image_011: loss=0.0002646006, psnr=35.774089, ssim=0.986071, lpips=0.016846
  image_012: loss=0.0009434924, psnr=30.252615, ssim=0.978801, lpips=0.019761
  image_013: loss=0.0006127786, psnr=32.126963, ssim=0.982687, lpips=0.015978
  image_014: loss=0.0005095097, psnr=32.928475, ssim=0.982413, lpips=0.015021
  image_015: loss=0.0004284970, psnr=33.680521, ssim=0.981391, lpips=0.017938
  image_016: loss=0.0003817236, psnr=34.182509, ssim=0.978988, lpips=0.018379
  image_017: loss=0.0004396533, psnr=33.568895, ssim=0.976395, lpips=0.027411
  image_018: loss=0.0003200645, psnr=34.947623, ssim=0.985360, lpips=0.012949
  image_019: loss=0.0004818221, psnr=33.171131, ssim=0.989298, lpips=0.012084
  image_020: loss=0.0005300876, psnr=32.756523, ssim=0.974107, lpips=0.037831
  image_021: loss=0.0005163340, psnr=32.870692, ssim=0.973504, lpips=0.039493
  image_022: loss=0.0005266858, psnr=32.784483, ssim=0.974401, lpips=0.032137
  image_023: loss=0.0005289585, psnr=32.765784, ssim=0.976682, lpips=0.028889
  image_024: loss=0.0003418729, psnr=34.661352, ssim=0.982387, lpips=0.022566
