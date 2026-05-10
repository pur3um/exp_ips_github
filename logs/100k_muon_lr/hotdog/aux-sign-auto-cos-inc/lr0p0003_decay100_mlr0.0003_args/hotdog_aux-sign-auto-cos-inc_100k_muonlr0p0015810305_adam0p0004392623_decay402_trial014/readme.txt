====================================================================================================
saved_time: 2026-05-02 18:18:12
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0015810305_adam0p0004392623_decay402_trial014 --optimizer aux-sign-auto-cos-inc --lrate 0.00043926227520300594 --lrate_decay 402 --muon_lrate 0.0015810304722280928 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0015810305_adam0p0004392623_decay402_trial014/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0015810305_adam0p0004392623_decay402_trial014
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 35 min
current_train_loss: 0.0010177487
current_train_psnr: 35.681690
testset_mean_loss: 0.0002235860
testset_mean_psnr: 37.271675
testset_mean_ssim: 0.981054
testset_mean_lpips: 0.016123
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001977251, psnr=37.039379, ssim=0.981315, lpips=0.015151
  image_001: loss=0.0001745916, psnr=37.579764, ssim=0.983679, lpips=0.016647
  image_002: loss=0.0002191699, psnr=36.592189, ssim=0.979448, lpips=0.020062
  image_003: loss=0.0001469224, psnr=38.329117, ssim=0.982480, lpips=0.018775
  image_004: loss=0.0001107402, psnr=39.556942, ssim=0.985231, lpips=0.014113
  image_005: loss=0.0001552724, psnr=38.089055, ssim=0.983700, lpips=0.011964
  image_006: loss=0.0002054662, psnr=36.872595, ssim=0.982806, lpips=0.012213
  image_007: loss=0.0001361273, psnr=38.660543, ssim=0.985380, lpips=0.011242
  image_008: loss=0.0001468883, psnr=38.330125, ssim=0.980667, lpips=0.012355
  image_009: loss=0.0001584665, psnr=38.000623, ssim=0.975692, lpips=0.019256
  image_010: loss=0.0001441550, psnr=38.411701, ssim=0.977166, lpips=0.021082
  image_011: loss=0.0002369778, psnr=36.252922, ssim=0.975162, lpips=0.021794
  image_012: loss=0.0002334968, psnr=36.317188, ssim=0.984019, lpips=0.011595
  image_013: loss=0.0001462685, psnr=38.348488, ssim=0.989212, lpips=0.008735
  image_014: loss=0.0002805439, psnr=35.519990, ssim=0.983339, lpips=0.015815
  image_015: loss=0.0011495451, psnr=29.394740, ssim=0.963663, lpips=0.030716
  image_016: loss=0.0004974980, psnr=33.032086, ssim=0.969354, lpips=0.028556
  image_017: loss=0.0001447486, psnr=38.393852, ssim=0.983098, lpips=0.015756
  image_018: loss=0.0002243955, psnr=36.489856, ssim=0.980577, lpips=0.012733
  image_019: loss=0.0001772840, psnr=37.513301, ssim=0.984037, lpips=0.010892
  image_020: loss=0.0001146912, psnr=39.404695, ssim=0.986806, lpips=0.010081
  image_021: loss=0.0001170651, psnr=39.315720, ssim=0.983874, lpips=0.011632
  image_022: loss=0.0001284419, psnr=38.912930, ssim=0.981918, lpips=0.018442
  image_023: loss=0.0001428407, psnr=38.451478, ssim=0.982249, lpips=0.018685
  image_024: loss=0.0002003276, psnr=36.982589, ssim=0.981464, lpips=0.014788
