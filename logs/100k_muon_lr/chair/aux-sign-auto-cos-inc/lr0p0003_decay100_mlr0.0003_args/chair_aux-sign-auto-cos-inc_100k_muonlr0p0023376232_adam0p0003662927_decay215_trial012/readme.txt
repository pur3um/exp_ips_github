====================================================================================================
saved_time: 2026-05-01 04:13:58
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/chair.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/chair/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname chair_aux-sign-auto-cos-inc_100k_muonlr0p0023376232_adam0p0003662927_decay215_trial012 --optimizer aux-sign-auto-cos-inc --lrate 0.00036629270902562624 --lrate_decay 215 --muon_lrate 0.0023376231744957465 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/chair/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/chair_aux-sign-auto-cos-inc_100k_muonlr0p0023376232_adam0p0003662927_decay215_trial012/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: chair_aux-sign-auto-cos-inc_100k_muonlr0p0023376232_adam0p0003662927_decay215_trial012
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 6 min
current_train_loss: 0.0018583757
current_train_psnr: 32.161156
testset_mean_loss: 0.0003933667
testset_mean_psnr: 34.281768
testset_mean_ssim: 0.978812
testset_mean_lpips: 0.015219
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001357959, psnr=38.671130, ssim=0.993525, lpips=0.005956
  image_001: loss=0.0001907746, psnr=37.194792, ssim=0.989495, lpips=0.005688
  image_002: loss=0.0003974598, psnr=34.007067, ssim=0.980775, lpips=0.015799
  image_003: loss=0.0004158861, psnr=33.810255, ssim=0.978774, lpips=0.015805
  image_004: loss=0.0004594817, psnr=33.377317, ssim=0.977332, lpips=0.016155
  image_005: loss=0.0004958577, psnr=33.046428, ssim=0.975837, lpips=0.013991
  image_006: loss=0.0005034638, psnr=32.980317, ssim=0.973200, lpips=0.016367
  image_007: loss=0.0005388438, psnr=32.685370, ssim=0.969784, lpips=0.022234
  image_008: loss=0.0004024901, psnr=33.952447, ssim=0.975026, lpips=0.015046
  image_009: loss=0.0004376778, psnr=33.588454, ssim=0.975180, lpips=0.016560
  image_010: loss=0.0006014685, psnr=32.207870, ssim=0.972448, lpips=0.024623
  image_011: loss=0.0003935930, psnr=34.049525, ssim=0.979044, lpips=0.017139
  image_012: loss=0.0002803910, psnr=35.522358, ssim=0.983874, lpips=0.015066
  image_013: loss=0.0003316729, psnr=34.792899, ssim=0.981165, lpips=0.017640
  image_014: loss=0.0002609155, psnr=35.834999, ssim=0.984459, lpips=0.012840
  image_015: loss=0.0004267996, psnr=33.697759, ssim=0.977883, lpips=0.019343
  image_016: loss=0.0003280323, psnr=34.840832, ssim=0.982341, lpips=0.011972
  image_017: loss=0.0004742827, psnr=33.239626, ssim=0.977280, lpips=0.015378
  image_018: loss=0.0005088982, psnr=32.933690, ssim=0.975032, lpips=0.016224
  image_019: loss=0.0005106136, psnr=32.919076, ssim=0.971847, lpips=0.019023
  image_020: loss=0.0004429074, psnr=33.536870, ssim=0.972858, lpips=0.016111
  image_021: loss=0.0003646371, psnr=34.381390, ssim=0.976118, lpips=0.015111
  image_022: loss=0.0004002891, psnr=33.976262, ssim=0.975631, lpips=0.017428
  image_023: loss=0.0003340194, psnr=34.762282, ssim=0.981588, lpips=0.012938
  image_024: loss=0.0001979167, psnr=37.035173, ssim=0.989798, lpips=0.006049
