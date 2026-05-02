====================================================================================================
saved_time: 2026-05-01 09:11:14
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p000730836_adam0p0021399817_decay284_trial009 --optimizer aux-sign-auto-cos-inc --lrate 0.0021399816882859213 --lrate_decay 284 --muon_lrate 0.0007308359664212255 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p000730836_adam0p0021399817_decay284_trial009/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p000730836_adam0p0021399817_decay284_trial009
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 35 min
current_train_loss: 0.0014556296
current_train_psnr: 35.148251
testset_mean_loss: 0.0005167220
testset_mean_psnr: 33.041176
testset_mean_ssim: 0.977468
testset_mean_lpips: 0.027476
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004793121, psnr=33.193815, ssim=0.977461, lpips=0.017601
  image_001: loss=0.0003041740, psnr=35.168778, ssim=0.982818, lpips=0.022222
  image_002: loss=0.0005830126, psnr=32.343220, ssim=0.974023, lpips=0.031010
  image_003: loss=0.0004935138, psnr=33.067006, ssim=0.973747, lpips=0.032510
  image_004: loss=0.0004645802, psnr=33.329392, ssim=0.972424, lpips=0.031512
  image_005: loss=0.0004213693, psnr=33.753370, ssim=0.977429, lpips=0.030950
  image_006: loss=0.0004070163, psnr=33.903881, ssim=0.987809, lpips=0.015613
  image_007: loss=0.0004947032, psnr=33.056552, ssim=0.980436, lpips=0.018874
  image_008: loss=0.0005813856, psnr=32.355357, ssim=0.969332, lpips=0.046537
  image_009: loss=0.0004929540, psnr=33.071935, ssim=0.973297, lpips=0.028355
  image_010: loss=0.0003199750, psnr=34.948839, ssim=0.980693, lpips=0.021518
  image_011: loss=0.0003087750, psnr=35.103577, ssim=0.984274, lpips=0.021088
  image_012: loss=0.0011239221, psnr=29.492638, ssim=0.977380, lpips=0.022143
  image_013: loss=0.0006763419, psnr=31.698336, ssim=0.982070, lpips=0.015505
  image_014: loss=0.0005808171, psnr=32.359606, ssim=0.980929, lpips=0.018588
  image_015: loss=0.0004898895, psnr=33.099018, ssim=0.979623, lpips=0.022351
  image_016: loss=0.0004349429, psnr=33.615677, ssim=0.976712, lpips=0.022465
  image_017: loss=0.0004953581, psnr=33.050807, ssim=0.974966, lpips=0.030072
  image_018: loss=0.0003832419, psnr=34.165268, ssim=0.982373, lpips=0.017281
  image_019: loss=0.0006275698, psnr=32.023379, ssim=0.985905, lpips=0.015486
  image_020: loss=0.0006070453, psnr=32.167788, ssim=0.970350, lpips=0.046623
  image_021: loss=0.0005764082, psnr=32.392698, ssim=0.969184, lpips=0.050195
  image_022: loss=0.0005838105, psnr=32.337280, ssim=0.971089, lpips=0.039388
  image_023: loss=0.0006000647, psnr=32.218019, ssim=0.972455, lpips=0.040371
  image_024: loss=0.0003878674, psnr=34.113166, ssim=0.979917, lpips=0.028642
