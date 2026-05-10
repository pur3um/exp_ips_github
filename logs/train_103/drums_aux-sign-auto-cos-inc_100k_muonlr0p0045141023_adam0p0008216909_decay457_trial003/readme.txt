====================================================================================================
saved_time: 2026-05-02 05:05:34
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: run_ranksched_optims_optuna_ready.py --config configs/drums.txt --basedir logs/train_103 --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial003 --optimizer aux-sign-auto-cos-inc --lrate 0.0008216908785704095 --lrate_decay 457 --muon_lrate 0.004514102265895567 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 10000 --max_eval_views 2 --metric_out logs/train_103/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial003/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial003
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 30 min
current_train_loss: 0.0052243634
current_train_psnr: 28.252789
testset_mean_loss: 0.0029200078
testset_mean_psnr: 25.582327
testset_mean_ssim: 0.929519
testset_mean_lpips: 0.056511
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0017802317, psnr=27.495234, ssim=0.928509, lpips=0.052334
  image_001: loss=0.0021505053, psnr=26.674595, ssim=0.925300, lpips=0.058957
  image_002: loss=0.0013992054, psnr=28.541185, ssim=0.945210, lpips=0.043771
  image_003: loss=0.0023721682, psnr=26.248545, ssim=0.934771, lpips=0.040912
  image_004: loss=0.0019293956, psnr=27.145787, ssim=0.933526, lpips=0.042858
  image_005: loss=0.0018756416, psnr=27.268501, ssim=0.940669, lpips=0.049290
  image_006: loss=0.0042923377, psnr=23.673061, ssim=0.905647, lpips=0.077600
  image_007: loss=0.0035948253, psnr=24.443222, ssim=0.918860, lpips=0.074059
  image_008: loss=0.0030441205, psnr=25.165381, ssim=0.931366, lpips=0.050303
  image_009: loss=0.0036243475, psnr=24.407702, ssim=0.930420, lpips=0.053546
  image_010: loss=0.0033979989, psnr=24.687768, ssim=0.934937, lpips=0.055452
  image_011: loss=0.0039747600, psnr=24.006891, ssim=0.919003, lpips=0.069706
  image_012: loss=0.0036872833, psnr=24.332935, ssim=0.930198, lpips=0.062970
  image_013: loss=0.0027660991, psnr=25.581323, ssim=0.935712, lpips=0.054263
  image_014: loss=0.0025704647, psnr=25.899883, ssim=0.939727, lpips=0.054632
  image_015: loss=0.0037679672, psnr=24.238929, ssim=0.935768, lpips=0.046448
  image_016: loss=0.0020614460, psnr=26.858280, ssim=0.950657, lpips=0.038943
  image_017: loss=0.0029241124, psnr=25.340059, ssim=0.932233, lpips=0.058544
  image_018: loss=0.0031818771, psnr=24.973166, ssim=0.926964, lpips=0.062572
  image_019: loss=0.0052403817, psnr=22.806371, ssim=0.894345, lpips=0.078285
  image_020: loss=0.0035097257, psnr=24.547268, ssim=0.917072, lpips=0.066564
  image_021: loss=0.0037449221, psnr=24.265572, ssim=0.916632, lpips=0.059246
  image_022: loss=0.0024786466, psnr=26.057854, ssim=0.932469, lpips=0.056542
  image_023: loss=0.0020617051, psnr=26.857734, ssim=0.935955, lpips=0.055936
  image_024: loss=0.0015700262, psnr=28.040931, ssim=0.942020, lpips=0.049046
