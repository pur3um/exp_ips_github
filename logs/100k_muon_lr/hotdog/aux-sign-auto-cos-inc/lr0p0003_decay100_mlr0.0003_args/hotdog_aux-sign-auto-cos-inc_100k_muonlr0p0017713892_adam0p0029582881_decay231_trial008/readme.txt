====================================================================================================
saved_time: 2026-05-01 21:02:19
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0017713892_adam0p0029582881_decay231_trial008 --optimizer aux-sign-auto-cos-inc --lrate 0.0029582881396890246 --lrate_decay 231 --muon_lrate 0.0017713892125344971 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0017713892_adam0p0029582881_decay231_trial008/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0017713892_adam0p0029582881_decay231_trial008
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0009731959
current_train_psnr: 35.764782
testset_mean_loss: 0.0002308367
testset_mean_psnr: 37.282250
testset_mean_ssim: 0.981468
testset_mean_lpips: 0.015420
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001863309, psnr=37.297150, ssim=0.982043, lpips=0.014595
  image_001: loss=0.0001725069, psnr=37.631934, ssim=0.984253, lpips=0.016698
  image_002: loss=0.0002192537, psnr=36.590529, ssim=0.980158, lpips=0.017818
  image_003: loss=0.0001456948, psnr=38.365555, ssim=0.983176, lpips=0.017698
  image_004: loss=0.0001076870, psnr=39.678363, ssim=0.985609, lpips=0.012871
  image_005: loss=0.0001565413, psnr=38.053708, ssim=0.984162, lpips=0.011453
  image_006: loss=0.0002118032, psnr=36.740673, ssim=0.983181, lpips=0.011830
  image_007: loss=0.0001332640, psnr=38.752869, ssim=0.985898, lpips=0.010689
  image_008: loss=0.0001453421, psnr=38.376083, ssim=0.981140, lpips=0.012146
  image_009: loss=0.0001588884, psnr=37.989075, ssim=0.976504, lpips=0.017929
  image_010: loss=0.0001410945, psnr=38.504896, ssim=0.977948, lpips=0.019433
  image_011: loss=0.0002346694, psnr=36.295433, ssim=0.975797, lpips=0.021715
  image_012: loss=0.0002479788, psnr=36.055852, ssim=0.983609, lpips=0.012122
  image_013: loss=0.0001519152, psnr=38.183986, ssim=0.989221, lpips=0.008689
  image_014: loss=0.0002730374, psnr=35.637778, ssim=0.983412, lpips=0.015337
  image_015: loss=0.0013846785, psnr=28.586510, ssim=0.964242, lpips=0.034223
  image_016: loss=0.0004728304, psnr=33.252945, ssim=0.970059, lpips=0.026956
  image_017: loss=0.0001433131, psnr=38.437137, ssim=0.983376, lpips=0.015254
  image_018: loss=0.0002213235, psnr=36.549722, ssim=0.980648, lpips=0.011920
  image_019: loss=0.0001798180, psnr=37.451666, ssim=0.984075, lpips=0.010338
  image_020: loss=0.0001132883, psnr=39.458147, ssim=0.986831, lpips=0.008926
  image_021: loss=0.0001170869, psnr=39.314912, ssim=0.983916, lpips=0.010919
  image_022: loss=0.0001254760, psnr=39.014391, ssim=0.982638, lpips=0.016416
  image_023: loss=0.0001377014, psnr=38.610613, ssim=0.982600, lpips=0.016076
  image_024: loss=0.0001893951, psnr=37.226310, ssim=0.982195, lpips=0.013440
