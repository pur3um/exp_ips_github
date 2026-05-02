====================================================================================================
saved_time: 2026-04-30 08:21:41
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0016353358_adam0p0010615361_decay386_trial002 --optimizer aux-sign-auto-cos-inc --lrate 0.0010615360772719654 --lrate_decay 386 --muon_lrate 0.001635335754309247 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0016353358_adam0p0010615361_decay386_trial002/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0016353358_adam0p0010615361_decay386_trial002
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 36 min
current_train_loss: 0.0013055863
current_train_psnr: 34.577419
testset_mean_loss: 0.0004598973
testset_mean_psnr: 33.566400
testset_mean_ssim: 0.979822
testset_mean_lpips: 0.023137
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004609497, psnr=33.363463, ssim=0.978818, lpips=0.014026
  image_001: loss=0.0002770537, psnr=35.574358, ssim=0.984559, lpips=0.019105
  image_002: loss=0.0005398515, psnr=32.677256, ssim=0.977509, lpips=0.026677
  image_003: loss=0.0004132293, psnr=33.838087, ssim=0.977612, lpips=0.025560
  image_004: loss=0.0004224202, psnr=33.742552, ssim=0.975085, lpips=0.027093
  image_005: loss=0.0003837954, psnr=34.159001, ssim=0.979341, lpips=0.036032
  image_006: loss=0.0003217017, psnr=34.925466, ssim=0.990878, lpips=0.012465
  image_007: loss=0.0004238302, psnr=33.728080, ssim=0.982937, lpips=0.014950
  image_008: loss=0.0005411219, psnr=32.667048, ssim=0.971433, lpips=0.035308
  image_009: loss=0.0004349534, psnr=33.615572, ssim=0.975757, lpips=0.022323
  image_010: loss=0.0002857646, psnr=35.439914, ssim=0.982825, lpips=0.016136
  image_011: loss=0.0002668544, psnr=35.737254, ssim=0.985951, lpips=0.017640
  image_012: loss=0.0010417149, psnr=29.822511, ssim=0.977288, lpips=0.021515
  image_013: loss=0.0006260516, psnr=32.033898, ssim=0.982801, lpips=0.015389
  image_014: loss=0.0005079267, psnr=32.941989, ssim=0.982620, lpips=0.015771
  image_015: loss=0.0004349946, psnr=33.615160, ssim=0.981395, lpips=0.018844
  image_016: loss=0.0003867597, psnr=34.125586, ssim=0.978930, lpips=0.019429
  image_017: loss=0.0004391704, psnr=33.573668, ssim=0.976922, lpips=0.027476
  image_018: loss=0.0003191541, psnr=34.959995, ssim=0.984932, lpips=0.014039
  image_019: loss=0.0005201986, psnr=32.838307, ssim=0.988837, lpips=0.012968
  image_020: loss=0.0005161745, psnr=32.872033, ssim=0.973904, lpips=0.039430
  image_021: loss=0.0005209014, psnr=32.832444, ssim=0.972887, lpips=0.040608
  image_022: loss=0.0005295540, psnr=32.760896, ssim=0.973894, lpips=0.031474
  image_023: loss=0.0005391723, psnr=32.682723, ssim=0.976194, lpips=0.030959
  image_024: loss=0.0003441338, psnr=34.632725, ssim=0.982241, lpips=0.023208
