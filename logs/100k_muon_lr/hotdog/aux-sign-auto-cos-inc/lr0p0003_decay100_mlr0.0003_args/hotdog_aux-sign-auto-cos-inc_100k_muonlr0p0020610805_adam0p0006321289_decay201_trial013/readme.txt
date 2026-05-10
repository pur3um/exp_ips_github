====================================================================================================
saved_time: 2026-05-02 14:42:33
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0020610805_adam0p0006321289_decay201_trial013 --optimizer aux-sign-auto-cos-inc --lrate 0.0006321288935121226 --lrate_decay 201 --muon_lrate 0.0020610804662823373 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0020610805_adam0p0006321289_decay201_trial013/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0020610805_adam0p0006321289_decay201_trial013
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 37 min
current_train_loss: 0.0009618458
current_train_psnr: 35.939720
testset_mean_loss: 0.0002187273
testset_mean_psnr: 37.417448
testset_mean_ssim: 0.981891
testset_mean_lpips: 0.014720
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001889829, psnr=37.235773, ssim=0.982686, lpips=0.014091
  image_001: loss=0.0001718823, psnr=37.647685, ssim=0.984551, lpips=0.016058
  image_002: loss=0.0002124117, psnr=36.728214, ssim=0.980150, lpips=0.018074
  image_003: loss=0.0001451222, psnr=38.382659, ssim=0.982935, lpips=0.016037
  image_004: loss=0.0001088783, psnr=39.630583, ssim=0.985424, lpips=0.012881
  image_005: loss=0.0001518601, psnr=38.185560, ssim=0.984164, lpips=0.011192
  image_006: loss=0.0001960789, psnr=37.075689, ssim=0.983765, lpips=0.010978
  image_007: loss=0.0001292712, psnr=38.884977, ssim=0.986602, lpips=0.009856
  image_008: loss=0.0001429642, psnr=38.447724, ssim=0.982046, lpips=0.010746
  image_009: loss=0.0001614485, psnr=37.919657, ssim=0.976782, lpips=0.017945
  image_010: loss=0.0001387434, psnr=38.577874, ssim=0.978550, lpips=0.017889
  image_011: loss=0.0002215065, psnr=36.546134, ssim=0.976824, lpips=0.019803
  image_012: loss=0.0002246006, psnr=36.485888, ssim=0.984975, lpips=0.010260
  image_013: loss=0.0001388736, psnr=38.573801, ssim=0.989524, lpips=0.007836
  image_014: loss=0.0002583511, psnr=35.877895, ssim=0.983714, lpips=0.015976
  image_015: loss=0.0011973410, psnr=29.217821, ssim=0.964077, lpips=0.032947
  image_016: loss=0.0004826726, psnr=33.163473, ssim=0.970239, lpips=0.027198
  image_017: loss=0.0001404870, psnr=38.523636, ssim=0.983580, lpips=0.013918
  image_018: loss=0.0002098711, psnr=36.780472, ssim=0.981367, lpips=0.011667
  image_019: loss=0.0001763808, psnr=37.535486, ssim=0.984674, lpips=0.009961
  image_020: loss=0.0001112398, psnr=39.537395, ssim=0.987350, lpips=0.008657
  image_021: loss=0.0001149597, psnr=39.394541, ssim=0.984289, lpips=0.010435
  image_022: loss=0.0001231541, psnr=39.095508, ssim=0.983052, lpips=0.016275
  image_023: loss=0.0001363263, psnr=38.654202, ssim=0.983305, lpips=0.014447
  image_024: loss=0.0001847754, psnr=37.333556, ssim=0.982646, lpips=0.012869
