====================================================================================================
saved_time: 2026-05-03 02:22:31
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0005060778_adam0p0013155348_decay272_trial018 --optimizer aux-sign-auto-cos-inc --lrate 0.0013155347895750314 --lrate_decay 272 --muon_lrate 0.0005060777595551436 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0005060778_adam0p0013155348_decay272_trial018/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0005060778_adam0p0013155348_decay272_trial018
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 33 min
current_train_loss: 0.0027124416
current_train_psnr: 32.189487
testset_mean_loss: 0.0015245564
testset_mean_psnr: 28.321694
testset_mean_ssim: 0.958480
testset_mean_lpips: 0.033632
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0010592029, psnr=29.750208, ssim=0.963965, lpips=0.028194
  image_001: loss=0.0015773496, psnr=28.020720, ssim=0.957376, lpips=0.034084
  image_002: loss=0.0019524364, psnr=27.094231, ssim=0.951427, lpips=0.039144
  image_003: loss=0.0021028058, psnr=26.772008, ssim=0.946142, lpips=0.042118
  image_004: loss=0.0022426113, psnr=26.492460, ssim=0.947661, lpips=0.041525
  image_005: loss=0.0015859379, psnr=27.997138, ssim=0.957870, lpips=0.045487
  image_006: loss=0.0014282078, psnr=28.452086, ssim=0.962003, lpips=0.029956
  image_007: loss=0.0011362786, psnr=29.445151, ssim=0.966439, lpips=0.026256
  image_008: loss=0.0016147883, psnr=27.918844, ssim=0.952405, lpips=0.036885
  image_009: loss=0.0017672204, psnr=27.527092, ssim=0.949319, lpips=0.038615
  image_010: loss=0.0011239771, psnr=29.492425, ssim=0.966055, lpips=0.027304
  image_011: loss=0.0010143324, psnr=29.938197, ssim=0.971734, lpips=0.021321
  image_012: loss=0.0010093199, psnr=29.959711, ssim=0.971304, lpips=0.021901
  image_013: loss=0.0009524237, psnr=30.211697, ssim=0.971933, lpips=0.022037
  image_014: loss=0.0013929547, psnr=28.560630, ssim=0.961614, lpips=0.029030
  image_015: loss=0.0013653458, psnr=28.647573, ssim=0.956992, lpips=0.032975
  image_016: loss=0.0013772150, psnr=28.609982, ssim=0.966313, lpips=0.026123
  image_017: loss=0.0013509566, psnr=28.693586, ssim=0.964242, lpips=0.036274
  image_018: loss=0.0013083610, psnr=28.832724, ssim=0.964807, lpips=0.024869
  image_019: loss=0.0015713673, psnr=28.037223, ssim=0.959641, lpips=0.029014
  image_020: loss=0.0019350101, psnr=27.133167, ssim=0.950093, lpips=0.048604
  image_021: loss=0.0011545775, psnr=29.375769, ssim=0.958294, lpips=0.035970
  image_022: loss=0.0016900407, psnr=27.721028, ssim=0.948847, lpips=0.040696
  image_023: loss=0.0026781471, psnr=25.721656, ssim=0.944396, lpips=0.047542
  image_024: loss=0.0017230414, psnr=27.637043, ssim=0.951123, lpips=0.034871
