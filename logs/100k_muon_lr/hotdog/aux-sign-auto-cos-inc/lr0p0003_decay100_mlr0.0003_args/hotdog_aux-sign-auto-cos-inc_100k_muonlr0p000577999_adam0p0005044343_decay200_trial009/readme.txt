====================================================================================================
saved_time: 2026-05-02 00:32:07
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p000577999_adam0p0005044343_decay200_trial009 --optimizer aux-sign-auto-cos-inc --lrate 0.0005044342937808256 --lrate_decay 200 --muon_lrate 0.0005779990416750779 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p000577999_adam0p0005044343_decay200_trial009/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p000577999_adam0p0005044343_decay200_trial009
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0011730739
current_train_psnr: 34.775436
testset_mean_loss: 0.0002885660
testset_mean_psnr: 36.029233
testset_mean_ssim: 0.976165
testset_mean_lpips: 0.024959
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0002556772, psnr=35.923078, ssim=0.977054, lpips=0.023952
  image_001: loss=0.0002339206, psnr=36.309313, ssim=0.978912, lpips=0.025786
  image_002: loss=0.0002878333, psnr=35.408588, ssim=0.974986, lpips=0.032077
  image_003: loss=0.0002057845, psnr=36.865872, ssim=0.977297, lpips=0.027334
  image_004: loss=0.0001498385, psnr=38.243763, ssim=0.980887, lpips=0.021321
  image_005: loss=0.0002102228, psnr=36.773201, ssim=0.979441, lpips=0.019522
  image_006: loss=0.0002778937, psnr=35.561212, ssim=0.977718, lpips=0.019947
  image_007: loss=0.0001837852, psnr=37.356893, ssim=0.981441, lpips=0.017540
  image_008: loss=0.0001844502, psnr=37.341206, ssim=0.975530, lpips=0.020072
  image_009: loss=0.0002132765, psnr=36.710568, ssim=0.967794, lpips=0.033405
  image_010: loss=0.0001934319, psnr=37.134716, ssim=0.968953, lpips=0.035704
  image_011: loss=0.0003342824, psnr=34.758864, ssim=0.968494, lpips=0.033232
  image_012: loss=0.0003212355, psnr=34.931764, ssim=0.979623, lpips=0.017962
  image_013: loss=0.0002428960, psnr=36.145795, ssim=0.984750, lpips=0.013862
  image_014: loss=0.0003298471, psnr=34.816872, ssim=0.980015, lpips=0.020839
  image_015: loss=0.0013788702, psnr=28.604766, ssim=0.962075, lpips=0.034247
  image_016: loss=0.0005327759, psnr=32.734554, ssim=0.966793, lpips=0.031549
  image_017: loss=0.0002056323, psnr=36.869085, ssim=0.978713, lpips=0.021158
  image_018: loss=0.0002965269, psnr=35.279358, ssim=0.975272, lpips=0.018864
  image_019: loss=0.0002376607, psnr=36.240424, ssim=0.979341, lpips=0.017732
  image_020: loss=0.0001559390, psnr=38.070450, ssim=0.982427, lpips=0.019038
  image_021: loss=0.0001594402, psnr=37.974019, ssim=0.978140, lpips=0.023358
  image_022: loss=0.0001719797, psnr=37.645227, ssim=0.975652, lpips=0.033922
  image_023: loss=0.0001926194, psnr=37.152998, ssim=0.976466, lpips=0.032122
  image_024: loss=0.0002583308, psnr=35.878236, ssim=0.976359, lpips=0.029420
