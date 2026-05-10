====================================================================================================
saved_time: 2026-05-02 11:04:45
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0019810483_adam0p0015750528_decay326_trial012 --optimizer aux-sign-auto-cos-inc --lrate 0.0015750527540490304 --lrate_decay 326 --muon_lrate 0.0019810483107722482 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0019810483_adam0p0015750528_decay326_trial012/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0019810483_adam0p0015750528_decay326_trial012
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 31 min
current_train_loss: 0.0009100201
current_train_psnr: 35.722633
testset_mean_loss: 0.0002251813
testset_mean_psnr: 37.386600
testset_mean_ssim: 0.981798
testset_mean_lpips: 0.014674
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001963278, psnr=37.070180, ssim=0.982275, lpips=0.014216
  image_001: loss=0.0001718148, psnr=37.649392, ssim=0.984487, lpips=0.015708
  image_002: loss=0.0002123031, psnr=36.730435, ssim=0.980321, lpips=0.016891
  image_003: loss=0.0001433252, psnr=38.436771, ssim=0.983440, lpips=0.015933
  image_004: loss=0.0001089642, psnr=39.627157, ssim=0.985559, lpips=0.012510
  image_005: loss=0.0001513753, psnr=38.199447, ssim=0.984371, lpips=0.010609
  image_006: loss=0.0001938771, psnr=37.124733, ssim=0.983834, lpips=0.010608
  image_007: loss=0.0001308716, psnr=38.831542, ssim=0.986329, lpips=0.010331
  image_008: loss=0.0001437168, psnr=38.424921, ssim=0.981895, lpips=0.011140
  image_009: loss=0.0001548826, psnr=38.099970, ssim=0.977028, lpips=0.017952
  image_010: loss=0.0001338078, psnr=38.735184, ssim=0.978938, lpips=0.019030
  image_011: loss=0.0002207411, psnr=36.561166, ssim=0.976662, lpips=0.020420
  image_012: loss=0.0002174387, psnr=36.626630, ssim=0.984859, lpips=0.010526
  image_013: loss=0.0001414415, psnr=38.494229, ssim=0.989469, lpips=0.007873
  image_014: loss=0.0002634602, psnr=35.792848, ssim=0.983823, lpips=0.015155
  image_015: loss=0.0012812050, psnr=28.923813, ssim=0.963095, lpips=0.031683
  image_016: loss=0.0005636827, psnr=32.489652, ssim=0.968673, lpips=0.027434
  image_017: loss=0.0001414852, psnr=38.492887, ssim=0.983509, lpips=0.014371
  image_018: loss=0.0002138480, psnr=36.698945, ssim=0.980981, lpips=0.011604
  image_019: loss=0.0001711745, psnr=37.665606, ssim=0.984854, lpips=0.009696
  image_020: loss=0.0001124650, psnr=39.489823, ssim=0.987431, lpips=0.008728
  image_021: loss=0.0001112798, psnr=39.535833, ssim=0.984530, lpips=0.009923
  image_022: loss=0.0001230905, psnr=39.097751, ssim=0.983081, lpips=0.015796
  image_023: loss=0.0001348545, psnr=38.701341, ssim=0.983168, lpips=0.015372
  image_024: loss=0.0001920990, psnr=37.164746, ssim=0.982328, lpips=0.013347
