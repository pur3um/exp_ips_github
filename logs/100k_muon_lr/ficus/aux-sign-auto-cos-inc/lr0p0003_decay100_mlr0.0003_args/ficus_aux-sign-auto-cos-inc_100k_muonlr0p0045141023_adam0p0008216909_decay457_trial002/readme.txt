====================================================================================================
saved_time: 2026-04-30 18:29:24
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial002 --optimizer aux-sign-auto-cos-inc --lrate 0.0008216908785704095 --lrate_decay 457 --muon_lrate 0.004514102265895567 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial002/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial002
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 27 min
current_train_loss: 0.0028562730
current_train_psnr: 32.491573
testset_mean_loss: 0.0013365403
testset_mean_psnr: 28.866570
testset_mean_ssim: 0.963816
testset_mean_lpips: 0.024652
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009482246, psnr=30.230888, ssim=0.967954, lpips=0.022326
  image_001: loss=0.0014369496, psnr=28.425584, ssim=0.962179, lpips=0.028728
  image_002: loss=0.0017341526, psnr=27.609127, ssim=0.957240, lpips=0.028803
  image_003: loss=0.0018635746, psnr=27.296532, ssim=0.953198, lpips=0.028894
  image_004: loss=0.0020550571, psnr=26.871761, ssim=0.952872, lpips=0.034377
  image_005: loss=0.0013992704, psnr=28.540983, ssim=0.963128, lpips=0.026688
  image_006: loss=0.0014350184, psnr=28.431425, ssim=0.964781, lpips=0.025503
  image_007: loss=0.0009803518, psnr=30.086180, ssim=0.971081, lpips=0.017346
  image_008: loss=0.0014207353, psnr=28.474868, ssim=0.959080, lpips=0.027180
  image_009: loss=0.0015450715, psnr=28.110514, ssim=0.955684, lpips=0.027392
  image_010: loss=0.0010045130, psnr=29.980444, ssim=0.971528, lpips=0.018650
  image_011: loss=0.0009001732, psnr=30.456739, ssim=0.976255, lpips=0.015712
  image_012: loss=0.0009109918, psnr=30.404855, ssim=0.975496, lpips=0.016869
  image_013: loss=0.0008627313, psnr=30.641244, ssim=0.976179, lpips=0.016930
  image_014: loss=0.0012751392, psnr=28.944424, ssim=0.966719, lpips=0.023284
  image_015: loss=0.0011930296, psnr=29.233488, ssim=0.963515, lpips=0.026023
  image_016: loss=0.0012891826, psnr=28.896855, ssim=0.970438, lpips=0.018112
  image_017: loss=0.0012034347, psnr=29.195774, ssim=0.969647, lpips=0.022931
  image_018: loss=0.0011933806, psnr=29.232210, ssim=0.967534, lpips=0.021548
  image_019: loss=0.0013869355, psnr=28.579437, ssim=0.964899, lpips=0.023542
  image_020: loss=0.0016187268, psnr=27.908264, ssim=0.956930, lpips=0.030540
  image_021: loss=0.0009829176, psnr=30.074828, ssim=0.964071, lpips=0.024937
  image_022: loss=0.0014796654, psnr=28.298365, ssim=0.954917, lpips=0.030506
  image_023: loss=0.0018613315, psnr=27.301762, ssim=0.952990, lpips=0.032794
  image_024: loss=0.0014329500, psnr=28.437689, ssim=0.957090, lpips=0.026694
