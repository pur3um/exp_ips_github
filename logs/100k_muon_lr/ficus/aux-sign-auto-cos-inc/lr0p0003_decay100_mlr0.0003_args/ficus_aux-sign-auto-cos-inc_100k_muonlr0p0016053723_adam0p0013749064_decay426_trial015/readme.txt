====================================================================================================
saved_time: 2026-05-02 15:42:35
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0016053723_adam0p0013749064_decay426_trial015 --optimizer aux-sign-auto-cos-inc --lrate 0.001374906384778521 --lrate_decay 426 --muon_lrate 0.0016053722586884453 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0016053723_adam0p0013749064_decay426_trial015/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0016053723_adam0p0013749064_decay426_trial015
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 33 min
current_train_loss: 0.0020731098
current_train_psnr: 33.780685
testset_mean_loss: 0.0012181452
testset_mean_psnr: 29.296085
testset_mean_ssim: 0.966632
testset_mean_lpips: 0.023604
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008687064, psnr=30.611269, ssim=0.970502, lpips=0.021577
  image_001: loss=0.0013661674, psnr=28.644960, ssim=0.964273, lpips=0.027142
  image_002: loss=0.0015681394, psnr=28.046153, ssim=0.960774, lpips=0.028352
  image_003: loss=0.0017518677, psnr=27.564987, ssim=0.956404, lpips=0.028656
  image_004: loss=0.0017907767, psnr=27.469585, ssim=0.957972, lpips=0.030780
  image_005: loss=0.0012137820, psnr=29.158593, ssim=0.967451, lpips=0.025965
  image_006: loss=0.0011983714, psnr=29.214085, ssim=0.967776, lpips=0.022824
  image_007: loss=0.0008318326, psnr=30.799640, ssim=0.974568, lpips=0.017656
  image_008: loss=0.0012865770, psnr=28.905642, ssim=0.961460, lpips=0.027891
  image_009: loss=0.0015317904, psnr=28.148006, ssim=0.958263, lpips=0.025616
  image_010: loss=0.0009159303, psnr=30.381375, ssim=0.973964, lpips=0.016423
  image_011: loss=0.0008185554, psnr=30.869519, ssim=0.978096, lpips=0.015464
  image_012: loss=0.0008100033, psnr=30.915132, ssim=0.977749, lpips=0.015924
  image_013: loss=0.0007188933, psnr=31.433355, ssim=0.979325, lpips=0.013562
  image_014: loss=0.0011536648, psnr=29.379203, ssim=0.968806, lpips=0.021431
  image_015: loss=0.0010884614, psnr=29.631869, ssim=0.965572, lpips=0.022833
  image_016: loss=0.0011355211, psnr=29.448048, ssim=0.973898, lpips=0.017126
  image_017: loss=0.0010573925, psnr=29.757638, ssim=0.972330, lpips=0.026996
  image_018: loss=0.0011781402, psnr=29.288030, ssim=0.969214, lpips=0.021523
  image_019: loss=0.0012125323, psnr=29.163067, ssim=0.967153, lpips=0.022432
  image_020: loss=0.0014638525, psnr=28.345027, ssim=0.960501, lpips=0.029094
  image_021: loss=0.0008250186, psnr=30.835362, ssim=0.970298, lpips=0.022634
  image_022: loss=0.0013710663, psnr=28.629415, ssim=0.959534, lpips=0.028223
  image_023: loss=0.0019004495, psnr=27.211436, ssim=0.951469, lpips=0.034411
  image_024: loss=0.0013961373, psnr=28.550718, ssim=0.958439, lpips=0.025559
