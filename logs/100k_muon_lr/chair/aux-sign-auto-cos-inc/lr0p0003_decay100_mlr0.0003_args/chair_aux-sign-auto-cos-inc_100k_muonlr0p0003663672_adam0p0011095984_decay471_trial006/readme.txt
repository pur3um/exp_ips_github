====================================================================================================
saved_time: 2026-04-30 15:21:54
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/chair.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/chair/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname chair_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial006 --optimizer aux-sign-auto-cos-inc --lrate 0.0011095983846308643 --lrate_decay 471 --muon_lrate 0.0003663671900803624 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/chair/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/chair_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial006/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: chair_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial006
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 6 min
current_train_loss: 0.0023108681
current_train_psnr: 30.534748
testset_mean_loss: 0.0006092581
testset_mean_psnr: 32.382763
testset_mean_ssim: 0.966357
testset_mean_lpips: 0.035701
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0002245345, psnr=36.487168, ssim=0.988393, lpips=0.013198
  image_001: loss=0.0003013852, psnr=35.208780, ssim=0.982313, lpips=0.014413
  image_002: loss=0.0005872675, psnr=32.311640, ssim=0.970175, lpips=0.031588
  image_003: loss=0.0006425388, psnr=31.921006, ssim=0.966370, lpips=0.037842
  image_004: loss=0.0007339374, psnr=31.343409, ssim=0.961904, lpips=0.042002
  image_005: loss=0.0008235045, psnr=30.843340, ssim=0.957846, lpips=0.044718
  image_006: loss=0.0008208673, psnr=30.857270, ssim=0.954852, lpips=0.047070
  image_007: loss=0.0008409425, psnr=30.752337, ssim=0.951511, lpips=0.051918
  image_008: loss=0.0006437213, psnr=31.913020, ssim=0.961109, lpips=0.040248
  image_009: loss=0.0006652229, psnr=31.770327, ssim=0.963288, lpips=0.032186
  image_010: loss=0.0008459687, psnr=30.726457, ssim=0.961080, lpips=0.040681
  image_011: loss=0.0005602166, psnr=32.516439, ssim=0.969516, lpips=0.037066
  image_012: loss=0.0004432556, psnr=33.533457, ssim=0.973125, lpips=0.034734
  image_013: loss=0.0004884924, psnr=33.111421, ssim=0.969734, lpips=0.036066
  image_014: loss=0.0004168068, psnr=33.800651, ssim=0.974798, lpips=0.029204
  image_015: loss=0.0005727942, psnr=32.420013, ssim=0.968669, lpips=0.034278
  image_016: loss=0.0005400396, psnr=32.675743, ssim=0.971124, lpips=0.030119
  image_017: loss=0.0007830051, psnr=31.062353, ssim=0.961285, lpips=0.038440
  image_018: loss=0.0008355928, psnr=30.780053, ssim=0.957216, lpips=0.045755
  image_019: loss=0.0008132086, psnr=30.897980, ssim=0.952914, lpips=0.046852
  image_020: loss=0.0007057859, psnr=31.513269, ssim=0.956241, lpips=0.047913
  image_021: loss=0.0005699644, psnr=32.441522, ssim=0.963466, lpips=0.040369
  image_022: loss=0.0005812268, psnr=32.356543, ssim=0.965771, lpips=0.033451
  image_023: loss=0.0004926564, psnr=33.074558, ssim=0.972813, lpips=0.027693
  image_024: loss=0.0002985162, psnr=35.250319, ssim=0.983419, lpips=0.014726
