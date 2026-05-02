====================================================================================================
saved_time: 2026-05-01 12:39:47
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p002353186_adam0p0004536829_decay104_trial010 --optimizer aux-sign-auto-cos-inc --lrate 0.000453682908579571 --lrate_decay 104 --muon_lrate 0.0023531859828563972 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p002353186_adam0p0004536829_decay104_trial010/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p002353186_adam0p0004536829_decay104_trial010
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 28 min
current_train_loss: 0.0014662590
current_train_psnr: 33.845699
testset_mean_loss: 0.0004603242
testset_mean_psnr: 33.582690
testset_mean_ssim: 0.980109
testset_mean_lpips: 0.022096
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004453271, psnr=33.513208, ssim=0.979184, lpips=0.015249
  image_001: loss=0.0002826006, psnr=35.488268, ssim=0.984525, lpips=0.018463
  image_002: loss=0.0004996050, psnr=33.013732, ssim=0.978346, lpips=0.024818
  image_003: loss=0.0004186993, psnr=33.780977, ssim=0.978095, lpips=0.025645
  image_004: loss=0.0004276536, psnr=33.689077, ssim=0.975211, lpips=0.026299
  image_005: loss=0.0003744370, psnr=34.266212, ssim=0.980140, lpips=0.025087
  image_006: loss=0.0003132641, psnr=35.040892, ssim=0.991206, lpips=0.011187
  image_007: loss=0.0004197383, psnr=33.770213, ssim=0.983454, lpips=0.013360
  image_008: loss=0.0005258799, psnr=32.791133, ssim=0.971605, lpips=0.036235
  image_009: loss=0.0004208928, psnr=33.758284, ssim=0.976554, lpips=0.023110
  image_010: loss=0.0002883215, psnr=35.401227, ssim=0.983183, lpips=0.015054
  image_011: loss=0.0002681951, psnr=35.715491, ssim=0.985882, lpips=0.017936
  image_012: loss=0.0011440518, psnr=29.415543, ssim=0.977125, lpips=0.020205
  image_013: loss=0.0006399320, psnr=31.938661, ssim=0.982995, lpips=0.015598
  image_014: loss=0.0005084003, psnr=32.937941, ssim=0.983190, lpips=0.015747
  image_015: loss=0.0004733437, psnr=33.248233, ssim=0.980938, lpips=0.018380
  image_016: loss=0.0003975778, psnr=34.005778, ssim=0.978598, lpips=0.019089
  image_017: loss=0.0004294793, psnr=33.670576, ssim=0.976826, lpips=0.027238
  image_018: loss=0.0003108220, psnr=35.074882, ssim=0.985456, lpips=0.011972
  image_019: loss=0.0004849384, psnr=33.143133, ssim=0.988927, lpips=0.011887
  image_020: loss=0.0005201271, psnr=32.838904, ssim=0.974166, lpips=0.037002
  image_021: loss=0.0005160823, psnr=32.872810, ssim=0.973247, lpips=0.040101
  image_022: loss=0.0005244756, psnr=32.802746, ssim=0.974556, lpips=0.032102
  image_023: loss=0.0005302117, psnr=32.755506, ssim=0.976656, lpips=0.029612
  image_024: loss=0.0003440476, psnr=34.633813, ssim=0.982657, lpips=0.021028
