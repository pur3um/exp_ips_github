====================================================================================================
saved_time: 2026-05-01 10:33:35
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0026909121_adam0p0003143806_decay155_trial005 --optimizer aux-sign-auto-cos-inc --lrate 0.0003143805912215675 --lrate_decay 155 --muon_lrate 0.0026909121249853807 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0026909121_adam0p0003143806_decay155_trial005/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0026909121_adam0p0003143806_decay155_trial005
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 30 min
current_train_loss: 0.0009387664
current_train_psnr: 36.288937
testset_mean_loss: 0.0002138080
testset_mean_psnr: 37.423504
testset_mean_ssim: 0.982070
testset_mean_lpips: 0.014427
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001911968, psnr=37.185192, ssim=0.982384, lpips=0.014190
  image_001: loss=0.0001718505, psnr=37.648489, ssim=0.984428, lpips=0.015453
  image_002: loss=0.0002131640, psnr=36.712859, ssim=0.980462, lpips=0.017368
  image_003: loss=0.0001440856, psnr=38.413792, ssim=0.983043, lpips=0.016434
  image_004: loss=0.0001106054, psnr=39.562233, ssim=0.985512, lpips=0.012832
  image_005: loss=0.0001516580, psnr=38.191344, ssim=0.984399, lpips=0.011014
  image_006: loss=0.0001946004, psnr=37.108560, ssim=0.983881, lpips=0.010912
  image_007: loss=0.0001316309, psnr=38.806418, ssim=0.986350, lpips=0.010440
  image_008: loss=0.0001346778, psnr=38.707036, ssim=0.982783, lpips=0.010670
  image_009: loss=0.0001557900, psnr=38.074600, ssim=0.978123, lpips=0.016410
  image_010: loss=0.0001354849, psnr=38.681089, ssim=0.979359, lpips=0.017500
  image_011: loss=0.0002238906, psnr=36.499640, ssim=0.977032, lpips=0.017718
  image_012: loss=0.0002413984, psnr=36.172655, ssim=0.984389, lpips=0.009760
  image_013: loss=0.0001512586, psnr=38.202795, ssim=0.989324, lpips=0.007631
  image_014: loss=0.0002782096, psnr=35.556277, ssim=0.983301, lpips=0.014489
  image_015: loss=0.0010732439, psnr=29.693015, ssim=0.965189, lpips=0.032463
  image_016: loss=0.0004426365, psnr=33.539527, ssim=0.971023, lpips=0.027354
  image_017: loss=0.0001404115, psnr=38.525971, ssim=0.983657, lpips=0.014222
  image_018: loss=0.0002133921, psnr=36.708214, ssim=0.981486, lpips=0.011531
  image_019: loss=0.0001741906, psnr=37.589750, ssim=0.984857, lpips=0.009231
  image_020: loss=0.0001125043, psnr=39.488303, ssim=0.987243, lpips=0.009559
  image_021: loss=0.0001126119, psnr=39.484152, ssim=0.984501, lpips=0.010021
  image_022: loss=0.0001227262, psnr=39.110624, ssim=0.983479, lpips=0.014947
  image_023: loss=0.0001358211, psnr=38.670323, ssim=0.983202, lpips=0.015146
  image_024: loss=0.0001881598, psnr=37.254729, ssim=0.982343, lpips=0.013369
