====================================================================================================
saved_time: 2026-05-02 09:00:48
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: run_ranksched_optims_optuna_ready.py --config configs/chair.txt --basedir logs/train_103 --expname chair_aux-sign-auto-cos-inc_100k_muonlr0p002664786_adam0p0029582881_decay230_trial010 --optimizer aux-sign-auto-cos-inc --lrate 0.002958288139689019 --lrate_decay 230 --muon_lrate 0.0026647860197380052 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out logs/train_103/chair/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/chair_aux-sign-auto-cos-inc_100k_muonlr0p002664786_adam0p0029582881_decay230_trial010/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: chair_aux-sign-auto-cos-inc_100k_muonlr0p002664786_adam0p0029582881_decay230_trial010
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 27 min
current_train_loss: 0.0016970494
current_train_psnr: 32.495674
testset_mean_loss: 0.0003840204
testset_mean_psnr: 34.384726
testset_mean_ssim: 0.979273
testset_mean_lpips: 0.014599
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001292872, psnr=38.884442, ssim=0.993729, lpips=0.005353
  image_001: loss=0.0001895526, psnr=37.222701, ssim=0.989582, lpips=0.005763
  image_002: loss=0.0003875828, psnr=34.116353, ssim=0.981042, lpips=0.017260
  image_003: loss=0.0004032134, psnr=33.944650, ssim=0.979617, lpips=0.015126
  image_004: loss=0.0004446733, psnr=33.519589, ssim=0.977797, lpips=0.016745
  image_005: loss=0.0004813326, psnr=33.175546, ssim=0.976526, lpips=0.013583
  image_006: loss=0.0004854033, psnr=33.138972, ssim=0.974052, lpips=0.014814
  image_007: loss=0.0005283314, psnr=32.770935, ssim=0.970499, lpips=0.020287
  image_008: loss=0.0003894112, psnr=34.095914, ssim=0.975832, lpips=0.014496
  image_009: loss=0.0004336056, psnr=33.629050, ssim=0.975396, lpips=0.015845
  image_010: loss=0.0005857732, psnr=32.322704, ssim=0.972924, lpips=0.025098
  image_011: loss=0.0003836704, psnr=34.160416, ssim=0.979240, lpips=0.016266
  image_012: loss=0.0002765225, psnr=35.582694, ssim=0.984020, lpips=0.014145
  image_013: loss=0.0003306438, psnr=34.806395, ssim=0.980969, lpips=0.016816
  image_014: loss=0.0002599267, psnr=35.851489, ssim=0.984429, lpips=0.012390
  image_015: loss=0.0004106273, psnr=33.865521, ssim=0.978550, lpips=0.019202
  image_016: loss=0.0003278587, psnr=34.843132, ssim=0.982380, lpips=0.011973
  image_017: loss=0.0004616882, psnr=33.356511, ssim=0.977817, lpips=0.015892
  image_018: loss=0.0005014362, psnr=32.997842, ssim=0.975507, lpips=0.015327
  image_019: loss=0.0005007306, psnr=33.003957, ssim=0.972252, lpips=0.016042
  image_020: loss=0.0004284678, psnr=33.680817, ssim=0.974027, lpips=0.014722
  image_021: loss=0.0003494229, psnr=34.566485, ssim=0.977063, lpips=0.014328
  image_022: loss=0.0003889224, psnr=34.101370, ssim=0.976544, lpips=0.016453
  image_023: loss=0.0003308935, psnr=34.803116, ssim=0.981897, lpips=0.011437
  image_024: loss=0.0001915332, psnr=37.177557, ssim=0.990143, lpips=0.005611
