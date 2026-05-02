====================================================================================================
saved_time: 2026-05-01 01:27:28
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial004 --optimizer aux-sign-auto-cos-inc --lrate 0.0011095983846308643 --lrate_decay 471 --muon_lrate 0.0003663671900803624 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial004/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial004
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 28 min
current_train_loss: 0.0028982034
current_train_psnr: 31.545839
testset_mean_loss: 0.0017172129
testset_mean_psnr: 27.793659
testset_mean_ssim: 0.953458
testset_mean_lpips: 0.038941
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0011852974, psnr=29.261726, ssim=0.959730, lpips=0.033038
  image_001: loss=0.0017614883, psnr=27.541202, ssim=0.952157, lpips=0.040931
  image_002: loss=0.0022385879, psnr=26.500258, ssim=0.945248, lpips=0.046014
  image_003: loss=0.0023786668, psnr=26.236664, ssim=0.940292, lpips=0.047304
  image_004: loss=0.0024640032, psnr=26.083587, ssim=0.941129, lpips=0.047166
  image_005: loss=0.0018774228, psnr=27.264379, ssim=0.951819, lpips=0.046455
  image_006: loss=0.0017122346, psnr=27.664367, ssim=0.957852, lpips=0.037275
  image_007: loss=0.0013562547, psnr=28.676587, ssim=0.960427, lpips=0.033680
  image_008: loss=0.0017953934, psnr=27.458403, ssim=0.946576, lpips=0.045547
  image_009: loss=0.0018880020, psnr=27.239975, ssim=0.944587, lpips=0.040661
  image_010: loss=0.0012579011, psnr=29.003535, ssim=0.960530, lpips=0.030584
  image_011: loss=0.0011404715, psnr=29.429155, ssim=0.966795, lpips=0.025492
  image_012: loss=0.0011411090, psnr=29.426728, ssim=0.966476, lpips=0.025383
  image_013: loss=0.0011141025, psnr=29.530748, ssim=0.966254, lpips=0.027486
  image_014: loss=0.0015174615, psnr=28.188823, ssim=0.957758, lpips=0.034839
  image_015: loss=0.0014716479, psnr=28.321961, ssim=0.952482, lpips=0.037290
  image_016: loss=0.0015411281, psnr=28.121612, ssim=0.961994, lpips=0.029033
  image_017: loss=0.0015406003, psnr=28.123100, ssim=0.960321, lpips=0.033934
  image_018: loss=0.0015751972, psnr=28.026650, ssim=0.960669, lpips=0.028682
  image_019: loss=0.0017296989, psnr=27.620295, ssim=0.955686, lpips=0.033956
  image_020: loss=0.0022263224, psnr=26.524119, ssim=0.943670, lpips=0.067103
  image_021: loss=0.0013965435, psnr=28.549455, ssim=0.951468, lpips=0.043166
  image_022: loss=0.0018758492, psnr=27.268021, ssim=0.944079, lpips=0.046897
  image_023: loss=0.0029695095, psnr=25.273153, ssim=0.940440, lpips=0.053558
  image_024: loss=0.0017754283, psnr=27.506968, ssim=0.948021, lpips=0.038053
