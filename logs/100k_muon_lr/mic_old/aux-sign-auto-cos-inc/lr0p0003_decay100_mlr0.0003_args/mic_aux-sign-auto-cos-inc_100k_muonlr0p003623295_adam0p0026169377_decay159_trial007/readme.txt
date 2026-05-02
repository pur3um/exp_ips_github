====================================================================================================
saved_time: 2026-05-01 02:05:03
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p003623295_adam0p0026169377_decay159_trial007 --optimizer aux-sign-auto-cos-inc --lrate 0.002616937682010607 --lrate_decay 159 --muon_lrate 0.003623295039993152 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p003623295_adam0p0026169377_decay159_trial007/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p003623295_adam0p0026169377_decay159_trial007
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 28 min
current_train_loss: 0.0018110471
current_train_psnr: 33.592800
testset_mean_loss: 0.0004628335
testset_mean_psnr: 33.531652
testset_mean_ssim: 0.979896
testset_mean_lpips: 0.022414
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004569923, psnr=33.400910, ssim=0.979059, lpips=0.014127
  image_001: loss=0.0002896021, psnr=35.381982, ssim=0.984429, lpips=0.017945
  image_002: loss=0.0004930708, psnr=33.070907, ssim=0.978651, lpips=0.023643
  image_003: loss=0.0004152311, psnr=33.817101, ssim=0.977972, lpips=0.025388
  image_004: loss=0.0004278541, psnr=33.687042, ssim=0.974997, lpips=0.026960
  image_005: loss=0.0003771776, psnr=34.234539, ssim=0.979562, lpips=0.027084
  image_006: loss=0.0003294684, psnr=34.821861, ssim=0.990739, lpips=0.011665
  image_007: loss=0.0004358119, psnr=33.607008, ssim=0.983274, lpips=0.013648
  image_008: loss=0.0005343612, psnr=32.721650, ssim=0.970755, lpips=0.038750
  image_009: loss=0.0004348986, psnr=33.616119, ssim=0.975399, lpips=0.023293
  image_010: loss=0.0003030487, psnr=35.184874, ssim=0.982433, lpips=0.014699
  image_011: loss=0.0002739013, psnr=35.624058, ssim=0.985826, lpips=0.016885
  image_012: loss=0.0010543213, psnr=29.770270, ssim=0.977912, lpips=0.022714
  image_013: loss=0.0006771740, psnr=31.692997, ssim=0.982039, lpips=0.017330
  image_014: loss=0.0005118105, psnr=32.908907, ssim=0.982617, lpips=0.016163
  image_015: loss=0.0004499166, psnr=33.468679, ssim=0.981554, lpips=0.018841
  image_016: loss=0.0003868917, psnr=34.124105, ssim=0.979093, lpips=0.018485
  image_017: loss=0.0004506794, psnr=33.461322, ssim=0.976431, lpips=0.026391
  image_018: loss=0.0003238287, psnr=34.896845, ssim=0.984877, lpips=0.013585
  image_019: loss=0.0004935455, psnr=33.066727, ssim=0.989133, lpips=0.012701
  image_020: loss=0.0005277559, psnr=32.775668, ssim=0.974044, lpips=0.036628
  image_021: loss=0.0005246702, psnr=32.801135, ssim=0.972505, lpips=0.041466
  image_022: loss=0.0005235942, psnr=32.810051, ssim=0.974811, lpips=0.031924
  image_023: loss=0.0005229647, psnr=32.815275, ssim=0.976695, lpips=0.029171
  image_024: loss=0.0003522672, psnr=34.531277, ssim=0.982587, lpips=0.020857
