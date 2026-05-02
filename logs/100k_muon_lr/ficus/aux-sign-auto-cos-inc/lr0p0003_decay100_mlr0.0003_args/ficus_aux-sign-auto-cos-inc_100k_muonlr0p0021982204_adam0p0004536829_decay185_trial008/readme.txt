====================================================================================================
saved_time: 2026-05-01 15:19:44
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0021982204_adam0p0004536829_decay185_trial008 --optimizer aux-sign-auto-cos-inc --lrate 0.000453682908579571 --lrate_decay 185 --muon_lrate 0.0021982203820131085 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0021982204_adam0p0004536829_decay185_trial008/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0021982204_adam0p0004536829_decay185_trial008
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 27 min
current_train_loss: 0.0021393569
current_train_psnr: 33.830940
testset_mean_loss: 0.0012363398
testset_mean_psnr: 29.248816
testset_mean_ssim: 0.966701
testset_mean_lpips: 0.023111
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008628840, psnr=30.640475, ssim=0.970688, lpips=0.021756
  image_001: loss=0.0013209343, psnr=28.791188, ssim=0.965276, lpips=0.026140
  image_002: loss=0.0015643222, psnr=28.056738, ssim=0.960950, lpips=0.025926
  image_003: loss=0.0017686811, psnr=27.523504, ssim=0.956369, lpips=0.028778
  image_004: loss=0.0017698753, psnr=27.520573, ssim=0.958897, lpips=0.030306
  image_005: loss=0.0012470474, psnr=29.041170, ssim=0.966371, lpips=0.024675
  image_006: loss=0.0013782288, psnr=28.606787, ssim=0.965048, lpips=0.025807
  image_007: loss=0.0008430097, psnr=30.741674, ssim=0.974809, lpips=0.017855
  image_008: loss=0.0012417641, psnr=29.059609, ssim=0.962372, lpips=0.027060
  image_009: loss=0.0015093333, psnr=28.212148, ssim=0.959377, lpips=0.025232
  image_010: loss=0.0009330995, psnr=30.300720, ssim=0.973378, lpips=0.017772
  image_011: loss=0.0007985666, psnr=30.976888, ssim=0.979033, lpips=0.014037
  image_012: loss=0.0008192639, psnr=30.865761, ssim=0.977496, lpips=0.015822
  image_013: loss=0.0007323328, psnr=31.352915, ssim=0.978668, lpips=0.015622
  image_014: loss=0.0011499797, psnr=29.393098, ssim=0.969241, lpips=0.019939
  image_015: loss=0.0011162421, psnr=29.522415, ssim=0.964936, lpips=0.024539
  image_016: loss=0.0011239493, psnr=29.492532, ssim=0.974047, lpips=0.016891
  image_017: loss=0.0010477157, psnr=29.797565, ssim=0.973326, lpips=0.020357
  image_018: loss=0.0011697785, psnr=29.318963, ssim=0.968936, lpips=0.020957
  image_019: loss=0.0012735217, psnr=28.949936, ssim=0.967177, lpips=0.023142
  image_020: loss=0.0014458318, psnr=28.398822, ssim=0.961062, lpips=0.027849
  image_021: loss=0.0007940956, psnr=31.001271, ssim=0.970546, lpips=0.020465
  image_022: loss=0.0013848194, psnr=28.586068, ssim=0.958615, lpips=0.026711
  image_023: loss=0.0021962398, psnr=26.583202, ssim=0.951807, lpips=0.034210
  image_024: loss=0.0014169789, psnr=28.486366, ssim=0.959112, lpips=0.025931
