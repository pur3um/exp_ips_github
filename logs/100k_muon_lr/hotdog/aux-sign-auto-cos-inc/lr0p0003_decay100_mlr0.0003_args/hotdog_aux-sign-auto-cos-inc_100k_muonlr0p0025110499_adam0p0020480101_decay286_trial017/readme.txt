====================================================================================================
saved_time: 2026-05-03 05:03:10
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0025110499_adam0p0020480101_decay286_trial017 --optimizer aux-sign-auto-cos-inc --lrate 0.002048010076638085 --lrate_decay 286 --muon_lrate 0.0025110498861300886 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0025110499_adam0p0020480101_decay286_trial017/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0025110499_adam0p0020480101_decay286_trial017
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 34 min
current_train_loss: 0.0009293342
current_train_psnr: 35.664116
testset_mean_loss: 0.0002232459
testset_mean_psnr: 37.417650
testset_mean_ssim: 0.981997
testset_mean_lpips: 0.014107
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001871790, psnr=37.277426, ssim=0.982590, lpips=0.013613
  image_001: loss=0.0001666589, psnr=37.781712, ssim=0.984681, lpips=0.013965
  image_002: loss=0.0002165398, psnr=36.644621, ssim=0.980398, lpips=0.017167
  image_003: loss=0.0001445481, psnr=38.399873, ssim=0.983118, lpips=0.016414
  image_004: loss=0.0001090347, psnr=39.624350, ssim=0.985774, lpips=0.011861
  image_005: loss=0.0001516913, psnr=38.190391, ssim=0.984232, lpips=0.010210
  image_006: loss=0.0002001817, psnr=36.985754, ssim=0.983571, lpips=0.011218
  image_007: loss=0.0001284244, psnr=38.913522, ssim=0.986620, lpips=0.010393
  image_008: loss=0.0001371396, psnr=38.628367, ssim=0.982359, lpips=0.010825
  image_009: loss=0.0001532225, psnr=38.146771, ssim=0.977758, lpips=0.016729
  image_010: loss=0.0001332702, psnr=38.752666, ssim=0.979502, lpips=0.017259
  image_011: loss=0.0002180305, psnr=36.614825, ssim=0.977007, lpips=0.018696
  image_012: loss=0.0002315678, psnr=36.353216, ssim=0.984336, lpips=0.010146
  image_013: loss=0.0001525581, psnr=38.165645, ssim=0.988982, lpips=0.008321
  image_014: loss=0.0002770787, psnr=35.573968, ssim=0.983210, lpips=0.015284
  image_015: loss=0.0013237494, psnr=28.781942, ssim=0.962999, lpips=0.028764
  image_016: loss=0.0004676276, psnr=33.300997, ssim=0.970939, lpips=0.026293
  image_017: loss=0.0001413892, psnr=38.495833, ssim=0.983709, lpips=0.014343
  image_018: loss=0.0002048830, psnr=36.884939, ssim=0.981586, lpips=0.010418
  image_019: loss=0.0001711558, psnr=37.666082, ssim=0.984972, lpips=0.009585
  image_020: loss=0.0001113747, psnr=39.532130, ssim=0.987471, lpips=0.008611
  image_021: loss=0.0001098604, psnr=39.591586, ssim=0.984917, lpips=0.009122
  image_022: loss=0.0001202849, psnr=39.197887, ssim=0.983458, lpips=0.014812
  image_023: loss=0.0001352465, psnr=38.688737, ssim=0.983516, lpips=0.014896
  image_024: loss=0.0001884512, psnr=37.248009, ssim=0.982214, lpips=0.013732
