====================================================================================================
saved_time: 2026-05-01 19:27:04
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p0025853902_adam0p0003505623_decay114_trial009 --optimizer aux-sign-auto-cos-inc --lrate 0.00035056225873487983 --lrate_decay 114 --muon_lrate 0.002585390206004178 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p0025853902_adam0p0003505623_decay114_trial009/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p0025853902_adam0p0003505623_decay114_trial009
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0023995014
current_train_psnr: 31.574438
testset_mean_loss: 0.0012292670
testset_mean_psnr: 29.522248
testset_mean_ssim: 0.877256
testset_mean_lpips: 0.085386
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009721820, psnr=30.122524, ssim=0.832055, lpips=0.125980
  image_001: loss=0.0011562428, psnr=29.369509, ssim=0.833595, lpips=0.128604
  image_002: loss=0.0009475109, psnr=30.234157, ssim=0.857214, lpips=0.110974
  image_003: loss=0.0013277752, psnr=28.768754, ssim=0.813268, lpips=0.137762
  image_004: loss=0.0010624612, psnr=29.736869, ssim=0.837482, lpips=0.102031
  image_005: loss=0.0008547042, psnr=30.681841, ssim=0.875866, lpips=0.082042
  image_006: loss=0.0012534709, psnr=29.018857, ssim=0.871148, lpips=0.084431
  image_007: loss=0.0010610122, psnr=29.742796, ssim=0.885466, lpips=0.075120
  image_008: loss=0.0009012156, psnr=30.451713, ssim=0.897731, lpips=0.066266
  image_009: loss=0.0008794209, psnr=30.558032, ssim=0.906851, lpips=0.056736
  image_010: loss=0.0008488140, psnr=30.711874, ssim=0.918892, lpips=0.045686
  image_011: loss=0.0019626834, psnr=27.071497, ssim=0.885055, lpips=0.072128
  image_012: loss=0.0030752795, psnr=25.121154, ssim=0.874408, lpips=0.090922
  image_013: loss=0.0035100675, psnr=24.546845, ssim=0.874375, lpips=0.082058
  image_014: loss=0.0014610726, psnr=28.353282, ssim=0.926268, lpips=0.050769
  image_015: loss=0.0009359284, psnr=30.287573, ssim=0.943402, lpips=0.040027
  image_016: loss=0.0008789125, psnr=30.560543, ssim=0.926126, lpips=0.045343
  image_017: loss=0.0009998265, psnr=30.000753, ssim=0.908393, lpips=0.054117
  image_018: loss=0.0015152692, psnr=28.195102, ssim=0.876020, lpips=0.082510
  image_019: loss=0.0011428199, psnr=29.420222, ssim=0.876110, lpips=0.084683
  image_020: loss=0.0008827853, psnr=30.541449, ssim=0.887296, lpips=0.083528
  image_021: loss=0.0008236771, psnr=30.842430, ssim=0.869848, lpips=0.092443
  image_022: loss=0.0006776772, psnr=31.689771, ssim=0.870287, lpips=0.095874
  image_023: loss=0.0006826067, psnr=31.658294, ssim=0.855529, lpips=0.117455
  image_024: loss=0.0009182589, psnr=30.370348, ssim=0.828724, lpips=0.127164
