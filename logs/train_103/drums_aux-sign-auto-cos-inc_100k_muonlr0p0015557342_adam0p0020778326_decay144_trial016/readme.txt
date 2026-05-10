====================================================================================================
saved_time: 2026-05-02 09:05:08
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: run_ranksched_optims_optuna_ready.py --config configs/drums.txt --basedir logs/train_103 --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0015557342_adam0p0020778326_decay144_trial016 --optimizer aux-sign-auto-cos-inc --lrate 0.0020778325842156463 --lrate_decay 144 --muon_lrate 0.0015557341979336068 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out logs/train_103/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0015557342_adam0p0020778326_decay144_trial016/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0015557342_adam0p0020778326_decay144_trial016
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 33 min
current_train_loss: 0.0044048447
current_train_psnr: 28.521858
testset_mean_loss: 0.0028947060
testset_mean_psnr: 25.688578
testset_mean_ssim: 0.929524
testset_mean_lpips: 0.057296
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0016647748, psnr=27.786445, ssim=0.927788, lpips=0.052585
  image_001: loss=0.0020420195, psnr=26.899401, ssim=0.925454, lpips=0.056945
  image_002: loss=0.0012579754, psnr=29.003278, ssim=0.949216, lpips=0.040885
  image_003: loss=0.0021597918, psnr=26.655881, ssim=0.934387, lpips=0.040203
  image_004: loss=0.0018826106, psnr=27.252395, ssim=0.934044, lpips=0.044914
  image_005: loss=0.0017411057, psnr=27.591748, ssim=0.942202, lpips=0.047182
  image_006: loss=0.0048639649, psnr=23.130096, ssim=0.903774, lpips=0.091418
  image_007: loss=0.0036692091, psnr=24.354275, ssim=0.916608, lpips=0.077101
  image_008: loss=0.0028137208, psnr=25.507190, ssim=0.934921, lpips=0.047701
  image_009: loss=0.0036801151, psnr=24.341386, ssim=0.930801, lpips=0.055527
  image_010: loss=0.0033382513, psnr=24.764810, ssim=0.938346, lpips=0.049924
  image_011: loss=0.0041786386, psnr=23.789652, ssim=0.919787, lpips=0.072808
  image_012: loss=0.0037823173, psnr=24.222420, ssim=0.928324, lpips=0.061610
  image_013: loss=0.0026041523, psnr=25.843336, ssim=0.938336, lpips=0.054310
  image_014: loss=0.0024018176, psnr=26.194600, ssim=0.942749, lpips=0.055469
  image_015: loss=0.0040678987, psnr=23.906299, ssim=0.933355, lpips=0.049961
  image_016: loss=0.0019739063, psnr=27.046734, ssim=0.952450, lpips=0.040835
  image_017: loss=0.0028839447, psnr=25.400131, ssim=0.931765, lpips=0.060545
  image_018: loss=0.0029316721, psnr=25.328846, ssim=0.929720, lpips=0.061971
  image_019: loss=0.0055038016, psnr=22.593372, ssim=0.887137, lpips=0.087128
  image_020: loss=0.0032715693, psnr=24.852439, ssim=0.917048, lpips=0.068320
  image_021: loss=0.0037986562, psnr=24.203700, ssim=0.913910, lpips=0.059741
  image_022: loss=0.0023518601, psnr=26.285885, ssim=0.931597, lpips=0.056774
  image_023: loss=0.0020533763, psnr=26.875314, ssim=0.933759, lpips=0.052702
  image_024: loss=0.0014505005, psnr=28.384821, ssim=0.940624, lpips=0.045840
