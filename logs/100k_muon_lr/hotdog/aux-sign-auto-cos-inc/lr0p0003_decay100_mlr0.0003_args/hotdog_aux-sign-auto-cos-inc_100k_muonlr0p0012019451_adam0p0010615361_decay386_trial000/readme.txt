====================================================================================================
saved_time: 2026-04-30 17:01:15
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0012019451_adam0p0010615361_decay386_trial000 --optimizer aux-sign-auto-cos-inc --lrate 0.0010615360772719654 --lrate_decay 386 --muon_lrate 0.0012019450985893186 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0012019451_adam0p0010615361_decay386_trial000/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0012019451_adam0p0010615361_decay386_trial000
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 30 min
current_train_loss: 0.0009930534
current_train_psnr: 35.460934
testset_mean_loss: 0.0002340775
testset_mean_psnr: 37.044751
testset_mean_ssim: 0.980472
testset_mean_lpips: 0.018089
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001992064, psnr=37.006966, ssim=0.981251, lpips=0.015731
  image_001: loss=0.0001812644, psnr=37.416872, ssim=0.983073, lpips=0.020885
  image_002: loss=0.0002288957, psnr=36.403621, ssim=0.978637, lpips=0.025146
  image_003: loss=0.0001543811, psnr=38.114055, ssim=0.982015, lpips=0.018502
  image_004: loss=0.0001166042, psnr=39.332853, ssim=0.984749, lpips=0.014106
  image_005: loss=0.0001655498, psnr=37.810710, ssim=0.983148, lpips=0.012668
  image_006: loss=0.0002188599, psnr=36.598335, ssim=0.981915, lpips=0.012910
  image_007: loss=0.0001390662, psnr=38.567780, ssim=0.985271, lpips=0.012689
  image_008: loss=0.0001568733, psnr=38.044506, ssim=0.980227, lpips=0.013271
  image_009: loss=0.0001738001, psnr=37.599498, ssim=0.974374, lpips=0.022157
  image_010: loss=0.0001519800, psnr=38.182132, ssim=0.975910, lpips=0.023515
  image_011: loss=0.0002494545, psnr=36.030086, ssim=0.974759, lpips=0.023260
  image_012: loss=0.0002506463, psnr=36.009385, ssim=0.983355, lpips=0.011477
  image_013: loss=0.0001587225, psnr=37.993612, ssim=0.988936, lpips=0.009396
  image_014: loss=0.0002819975, psnr=35.497545, ssim=0.983211, lpips=0.016591
  image_015: loss=0.0011948118, psnr=29.227005, ssim=0.965418, lpips=0.032504
  image_016: loss=0.0004846571, psnr=33.145654, ssim=0.969558, lpips=0.027838
  image_017: loss=0.0001558747, psnr=38.072240, ssim=0.982449, lpips=0.016560
  image_018: loss=0.0002478553, psnr=36.058016, ssim=0.979092, lpips=0.013632
  image_019: loss=0.0001907912, psnr=37.194415, ssim=0.983233, lpips=0.011701
  image_020: loss=0.0001200025, psnr=39.208094, ssim=0.986239, lpips=0.011626
  image_021: loss=0.0001233444, psnr=39.088800, ssim=0.982653, lpips=0.013838
  image_022: loss=0.0001325181, psnr=38.777244, ssim=0.980852, lpips=0.021628
  image_023: loss=0.0001481928, psnr=38.291725, ssim=0.981260, lpips=0.021084
  image_024: loss=0.0002265875, psnr=36.447640, ssim=0.980206, lpips=0.029508
