====================================================================================================
saved_time: 2026-05-03 08:38:16
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0022236206_adam0p0014265056_decay243_trial018 --optimizer aux-sign-auto-cos-inc --lrate 0.0014265056027087768 --lrate_decay 243 --muon_lrate 0.0022236205726516934 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0022236206_adam0p0014265056_decay243_trial018/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0022236206_adam0p0014265056_decay243_trial018
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 34 min
current_train_loss: 0.0009403116
current_train_psnr: 35.679783
testset_mean_loss: 0.0002359098
testset_mean_psnr: 37.356789
testset_mean_ssim: 0.981880
testset_mean_lpips: 0.014770
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001851068, psnr=37.325774, ssim=0.982460, lpips=0.014417
  image_001: loss=0.0001726043, psnr=37.629482, ssim=0.984381, lpips=0.015643
  image_002: loss=0.0002202468, psnr=36.570902, ssim=0.979892, lpips=0.018339
  image_003: loss=0.0001396965, psnr=38.548143, ssim=0.983539, lpips=0.015891
  image_004: loss=0.0001070747, psnr=39.703128, ssim=0.985790, lpips=0.012922
  image_005: loss=0.0001512494, psnr=38.203061, ssim=0.984508, lpips=0.011013
  image_006: loss=0.0001973438, psnr=37.047763, ssim=0.983796, lpips=0.010616
  image_007: loss=0.0001301967, psnr=38.853995, ssim=0.986473, lpips=0.009701
  image_008: loss=0.0001391646, psnr=38.564709, ssim=0.981844, lpips=0.010960
  image_009: loss=0.0001590649, psnr=37.984254, ssim=0.977163, lpips=0.017078
  image_010: loss=0.0001340652, psnr=38.726836, ssim=0.979087, lpips=0.017133
  image_011: loss=0.0002253205, psnr=36.471990, ssim=0.976447, lpips=0.018929
  image_012: loss=0.0002323158, psnr=36.339212, ssim=0.984111, lpips=0.010362
  image_013: loss=0.0001395905, psnr=38.551439, ssim=0.989571, lpips=0.007567
  image_014: loss=0.0002746162, psnr=35.612737, ssim=0.983607, lpips=0.015095
  image_015: loss=0.0015870467, psnr=27.994103, ssim=0.963315, lpips=0.032281
  image_016: loss=0.0005015084, psnr=32.997217, ssim=0.970887, lpips=0.026970
  image_017: loss=0.0001395310, psnr=38.553289, ssim=0.983634, lpips=0.014299
  image_018: loss=0.0002139934, psnr=36.695993, ssim=0.981298, lpips=0.011408
  image_019: loss=0.0001756628, psnr=37.553199, ssim=0.984739, lpips=0.009346
  image_020: loss=0.0001123418, psnr=39.494582, ssim=0.987271, lpips=0.008736
  image_021: loss=0.0001137160, psnr=39.441781, ssim=0.984423, lpips=0.010511
  image_022: loss=0.0001232666, psnr=39.091541, ssim=0.983164, lpips=0.015093
  image_023: loss=0.0001339220, psnr=38.731478, ssim=0.983450, lpips=0.015229
  image_024: loss=0.0001890987, psnr=37.233113, ssim=0.982142, lpips=0.019702
