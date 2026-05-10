====================================================================================================
saved_time: 2026-05-03 05:54:17
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0025778959_adam0p0023051783_decay338_trial019 --optimizer aux-sign-auto-cos-inc --lrate 0.0023051783092912963 --lrate_decay 338 --muon_lrate 0.002577895857881722 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0025778959_adam0p0023051783_decay338_trial019/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0025778959_adam0p0023051783_decay338_trial019
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 31 min
current_train_loss: 0.0022750676
current_train_psnr: 31.960693
testset_mean_loss: 0.0012045191
testset_mean_psnr: 29.324763
testset_mean_ssim: 0.967176
testset_mean_lpips: 0.022271
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008385789, psnr=30.764560, ssim=0.970962, lpips=0.021747
  image_001: loss=0.0013137900, psnr=28.814740, ssim=0.965436, lpips=0.022586
  image_002: loss=0.0015514963, psnr=28.092492, ssim=0.962110, lpips=0.027373
  image_003: loss=0.0017012099, psnr=27.692421, ssim=0.957637, lpips=0.028017
  image_004: loss=0.0017533506, psnr=27.561312, ssim=0.958622, lpips=0.029478
  image_005: loss=0.0012433655, psnr=29.054012, ssim=0.967038, lpips=0.023282
  image_006: loss=0.0012600973, psnr=28.995959, ssim=0.967035, lpips=0.022667
  image_007: loss=0.0008495825, psnr=30.707944, ssim=0.974410, lpips=0.015676
  image_008: loss=0.0012875026, psnr=28.902519, ssim=0.962499, lpips=0.027163
  image_009: loss=0.0014606721, psnr=28.354472, ssim=0.959146, lpips=0.025748
  image_010: loss=0.0009292004, psnr=30.318905, ssim=0.973429, lpips=0.017199
  image_011: loss=0.0008351171, psnr=30.782526, ssim=0.978303, lpips=0.015161
  image_012: loss=0.0008135159, psnr=30.896339, ssim=0.977650, lpips=0.014846
  image_013: loss=0.0007344796, psnr=31.340202, ssim=0.979256, lpips=0.014173
  image_014: loss=0.0011646861, psnr=29.337911, ssim=0.969844, lpips=0.019483
  image_015: loss=0.0011012158, psnr=29.581275, ssim=0.965793, lpips=0.023026
  image_016: loss=0.0011171725, psnr=29.518797, ssim=0.974398, lpips=0.017015
  image_017: loss=0.0010576524, psnr=29.756570, ssim=0.972441, lpips=0.019181
  image_018: loss=0.0011229361, psnr=29.496449, ssim=0.969873, lpips=0.020109
  image_019: loss=0.0013286980, psnr=28.765737, ssim=0.966166, lpips=0.023312
  image_020: loss=0.0014208945, psnr=28.474381, ssim=0.962037, lpips=0.026380
  image_021: loss=0.0008295548, psnr=30.811549, ssim=0.969683, lpips=0.021424
  image_022: loss=0.0013473821, psnr=28.705092, ssim=0.959093, lpips=0.028093
  image_023: loss=0.0017050569, psnr=27.682611, ssim=0.956230, lpips=0.028567
  image_024: loss=0.0013457687, psnr=28.710295, ssim=0.960310, lpips=0.025059
