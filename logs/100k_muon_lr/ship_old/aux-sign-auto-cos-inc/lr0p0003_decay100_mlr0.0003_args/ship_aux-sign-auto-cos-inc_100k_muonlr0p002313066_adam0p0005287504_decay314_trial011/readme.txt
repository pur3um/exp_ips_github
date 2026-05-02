====================================================================================================
saved_time: 2026-05-02 02:27:35
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p002313066_adam0p0005287504_decay314_trial011 --optimizer aux-sign-auto-cos-inc --lrate 0.000528750432764932 --lrate_decay 314 --muon_lrate 0.0023130659551536155 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p002313066_adam0p0005287504_decay314_trial011/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p002313066_adam0p0005287504_decay314_trial011
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0024932055
current_train_psnr: 31.916410
testset_mean_loss: 0.0011679848
testset_mean_psnr: 29.621267
testset_mean_ssim: 0.877464
testset_mean_lpips: 0.084803
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009679851, psnr=30.141313, ssim=0.833182, lpips=0.126066
  image_001: loss=0.0011478696, psnr=29.401074, ssim=0.834445, lpips=0.130168
  image_002: loss=0.0009319756, psnr=30.305954, ssim=0.857748, lpips=0.111423
  image_003: loss=0.0012863231, psnr=28.906499, ssim=0.813736, lpips=0.135947
  image_004: loss=0.0010409667, psnr=29.825631, ssim=0.837701, lpips=0.095689
  image_005: loss=0.0008640957, psnr=30.634381, ssim=0.875099, lpips=0.081110
  image_006: loss=0.0012467070, psnr=29.042356, ssim=0.871426, lpips=0.083382
  image_007: loss=0.0010374272, psnr=29.840423, ssim=0.886236, lpips=0.070743
  image_008: loss=0.0008893686, psnr=30.509182, ssim=0.898097, lpips=0.064607
  image_009: loss=0.0008731580, psnr=30.589071, ssim=0.907471, lpips=0.055667
  image_010: loss=0.0008199874, psnr=30.861927, ssim=0.920783, lpips=0.045813
  image_011: loss=0.0022239408, psnr=26.528768, ssim=0.882323, lpips=0.074266
  image_012: loss=0.0027697235, psnr=25.575636, ssim=0.868931, lpips=0.098721
  image_013: loss=0.0020757820, psnr=26.828182, ssim=0.890955, lpips=0.072345
  image_014: loss=0.0014717777, psnr=28.321578, ssim=0.926175, lpips=0.050790
  image_015: loss=0.0009298685, psnr=30.315784, ssim=0.945160, lpips=0.039710
  image_016: loss=0.0008688191, psnr=30.610706, ssim=0.926469, lpips=0.042975
  image_017: loss=0.0010311031, psnr=29.866979, ssim=0.903569, lpips=0.052338
  image_018: loss=0.0015689932, psnr=28.043789, ssim=0.871796, lpips=0.087220
  image_019: loss=0.0011374415, psnr=29.440709, ssim=0.875629, lpips=0.082582
  image_020: loss=0.0008951159, psnr=30.481207, ssim=0.885970, lpips=0.081395
  image_021: loss=0.0008215797, psnr=30.853502, ssim=0.869620, lpips=0.091028
  image_022: loss=0.0006827689, psnr=31.657262, ssim=0.869960, lpips=0.097266
  image_023: loss=0.0006849414, psnr=31.643465, ssim=0.855129, lpips=0.120037
  image_024: loss=0.0009319001, psnr=30.306306, ssim=0.828980, lpips=0.128778
