====================================================================================================
saved_time: 2026-05-03 08:09:49
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/ship.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p0025632393_adam0p0004840935_decay102_trial012 --optimizer aux-sign-auto-cos-inc --lrate 0.00048409351102394446 --lrate_decay 102 --muon_lrate 0.0025632393008060087 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p0025632393_adam0p0004840935_decay102_trial012/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p0025632393_adam0p0004840935_decay102_trial012
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 14 min
current_train_loss: 0.0025561592
current_train_psnr: 31.913748
testset_mean_loss: 0.0011601083
testset_mean_psnr: 29.634284
testset_mean_ssim: 0.878088
testset_mean_lpips: 0.084916
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009682376, psnr=30.140180, ssim=0.833304, lpips=0.127486
  image_001: loss=0.0011466357, psnr=29.405745, ssim=0.834708, lpips=0.129308
  image_002: loss=0.0009369855, psnr=30.282671, ssim=0.857777, lpips=0.112837
  image_003: loss=0.0013267379, psnr=28.772148, ssim=0.812673, lpips=0.139891
  image_004: loss=0.0010473789, psnr=29.798961, ssim=0.837250, lpips=0.103759
  image_005: loss=0.0008616950, psnr=30.646464, ssim=0.875881, lpips=0.083064
  image_006: loss=0.0012710541, psnr=28.958359, ssim=0.871314, lpips=0.084220
  image_007: loss=0.0010490861, psnr=29.791888, ssim=0.884968, lpips=0.071584
  image_008: loss=0.0008918757, psnr=30.496956, ssim=0.897956, lpips=0.064144
  image_009: loss=0.0008915274, psnr=30.498653, ssim=0.907376, lpips=0.056227
  image_010: loss=0.0008367342, psnr=30.774124, ssim=0.920498, lpips=0.045409
  image_011: loss=0.0017276466, psnr=27.625451, ssim=0.892594, lpips=0.066121
  image_012: loss=0.0029482120, psnr=25.304413, ssim=0.871193, lpips=0.092942
  image_013: loss=0.0021536646, psnr=26.668219, ssim=0.892021, lpips=0.071551
  image_014: loss=0.0014559845, psnr=28.368432, ssim=0.926500, lpips=0.050090
  image_015: loss=0.0009295644, psnr=30.317205, ssim=0.944722, lpips=0.041738
  image_016: loss=0.0008618797, psnr=30.645533, ssim=0.926081, lpips=0.045297
  image_017: loss=0.0010090555, psnr=29.960849, ssim=0.905944, lpips=0.054893
  image_018: loss=0.0014826423, psnr=28.289636, ssim=0.875754, lpips=0.081733
  image_019: loss=0.0011425087, psnr=29.421404, ssim=0.876236, lpips=0.082784
  image_020: loss=0.0009097485, psnr=30.410786, ssim=0.885435, lpips=0.082232
  image_021: loss=0.0008270501, psnr=30.824681, ssim=0.869873, lpips=0.091851
  image_022: loss=0.0006869941, psnr=31.630469, ssim=0.869013, lpips=0.098822
  image_023: loss=0.0006967341, psnr=31.569329, ssim=0.854728, lpips=0.118301
  image_024: loss=0.0009430732, psnr=30.254546, ssim=0.828395, lpips=0.126604
