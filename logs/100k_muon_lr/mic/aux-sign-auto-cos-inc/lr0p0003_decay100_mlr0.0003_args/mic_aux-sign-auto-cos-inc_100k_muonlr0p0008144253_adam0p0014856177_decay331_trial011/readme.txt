====================================================================================================
saved_time: 2026-05-01 16:09:34
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0008144253_adam0p0014856177_decay331_trial011 --optimizer aux-sign-auto-cos-inc --lrate 0.0014856177368056476 --lrate_decay 331 --muon_lrate 0.0008144252902731651 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0008144253_adam0p0014856177_decay331_trial011/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0008144253_adam0p0014856177_decay331_trial011
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0015533107
current_train_psnr: 33.931202
testset_mean_loss: 0.0005103856
testset_mean_psnr: 33.091919
testset_mean_ssim: 0.977433
testset_mean_lpips: 0.027470
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004761170, psnr=33.222863, ssim=0.977345, lpips=0.018520
  image_001: loss=0.0003025914, psnr=35.191432, ssim=0.982957, lpips=0.021789
  image_002: loss=0.0005590813, psnr=32.525249, ssim=0.975672, lpips=0.028323
  image_003: loss=0.0004770889, psnr=33.214006, ssim=0.974757, lpips=0.030016
  image_004: loss=0.0004664318, psnr=33.312117, ssim=0.972731, lpips=0.030580
  image_005: loss=0.0004238086, psnr=33.728302, ssim=0.977034, lpips=0.034147
  image_006: loss=0.0004092536, psnr=33.880074, ssim=0.987335, lpips=0.015737
  image_007: loss=0.0004928230, psnr=33.073090, ssim=0.979872, lpips=0.019531
  image_008: loss=0.0005829670, psnr=32.343560, ssim=0.968807, lpips=0.049514
  image_009: loss=0.0004954965, psnr=33.049593, ssim=0.973309, lpips=0.030086
  image_010: loss=0.0003232053, psnr=34.905214, ssim=0.980743, lpips=0.021259
  image_011: loss=0.0002964199, psnr=35.280925, ssim=0.984367, lpips=0.020506
  image_012: loss=0.0011063046, psnr=29.561252, ssim=0.976688, lpips=0.021394
  image_013: loss=0.0006640996, psnr=31.777667, ssim=0.981100, lpips=0.015867
  image_014: loss=0.0005414368, psnr=32.664522, ssim=0.982106, lpips=0.018884
  image_015: loss=0.0004823423, psnr=33.166446, ssim=0.979926, lpips=0.021469
  image_016: loss=0.0004327048, psnr=33.638082, ssim=0.977220, lpips=0.020498
  image_017: loss=0.0004831033, psnr=33.159599, ssim=0.974751, lpips=0.029533
  image_018: loss=0.0003732314, psnr=34.280217, ssim=0.982236, lpips=0.017096
  image_019: loss=0.0006059711, psnr=32.175480, ssim=0.984660, lpips=0.016215
  image_020: loss=0.0006046773, psnr=32.184763, ssim=0.970277, lpips=0.047399
  image_021: loss=0.0005775901, psnr=32.383802, ssim=0.969311, lpips=0.049518
  image_022: loss=0.0005831739, psnr=32.342018, ssim=0.971013, lpips=0.041090
  image_023: loss=0.0006096385, psnr=32.149276, ssim=0.972110, lpips=0.038555
  image_024: loss=0.0003900831, psnr=34.088427, ssim=0.979493, lpips=0.029219
