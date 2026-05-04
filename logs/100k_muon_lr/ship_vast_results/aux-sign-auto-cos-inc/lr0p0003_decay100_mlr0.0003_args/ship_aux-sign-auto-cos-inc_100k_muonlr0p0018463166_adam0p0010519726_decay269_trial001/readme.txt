====================================================================================================
saved_time: 2026-05-02 07:27:53
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/ship.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial001 --optimizer aux-sign-auto-cos-inc --lrate 0.0010519726230559959 --lrate_decay 269 --muon_lrate 0.001846316577722323 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial001/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p0018463166_adam0p0010519726_decay269_trial001
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 15 min
current_train_loss: 0.0023119601
current_train_psnr: 31.755560
testset_mean_loss: 0.0011468346
testset_mean_psnr: 29.657910
testset_mean_ssim: 0.877666
testset_mean_lpips: 0.085726
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009708873, psnr=30.128311, ssim=0.831481, lpips=0.127204
  image_001: loss=0.0011498374, psnr=29.393635, ssim=0.833755, lpips=0.130495
  image_002: loss=0.0009321063, psnr=30.305345, ssim=0.858821, lpips=0.115902
  image_003: loss=0.0013243558, psnr=28.779953, ssim=0.812963, lpips=0.137559
  image_004: loss=0.0010545780, psnr=29.769213, ssim=0.837669, lpips=0.098634
  image_005: loss=0.0008408861, psnr=30.752628, ssim=0.876520, lpips=0.085126
  image_006: loss=0.0012795199, psnr=28.929529, ssim=0.870115, lpips=0.088290
  image_007: loss=0.0010609889, psnr=29.742891, ssim=0.883212, lpips=0.075292
  image_008: loss=0.0008881519, psnr=30.515127, ssim=0.897975, lpips=0.068852
  image_009: loss=0.0008890665, psnr=30.510657, ssim=0.905554, lpips=0.056551
  image_010: loss=0.0008295165, psnr=30.811749, ssim=0.918993, lpips=0.046673
  image_011: loss=0.0018709425, psnr=27.279395, ssim=0.889187, lpips=0.067623
  image_012: loss=0.0025931546, psnr=25.861716, ssim=0.876671, lpips=0.088096
  image_013: loss=0.0020600401, psnr=26.861243, ssim=0.890144, lpips=0.069551
  image_014: loss=0.0014361059, psnr=28.428135, ssim=0.926522, lpips=0.050970
  image_015: loss=0.0009536977, psnr=30.205892, ssim=0.944277, lpips=0.041764
  image_016: loss=0.0008718455, psnr=30.595604, ssim=0.925871, lpips=0.044653
  image_017: loss=0.0010074417, psnr=29.967800, ssim=0.906190, lpips=0.055294
  image_018: loss=0.0014953373, psnr=28.252608, ssim=0.875539, lpips=0.085995
  image_019: loss=0.0011476325, psnr=29.401971, ssim=0.874384, lpips=0.085960
  image_020: loss=0.0008948155, psnr=30.482664, ssim=0.885181, lpips=0.084915
  image_021: loss=0.0008342312, psnr=30.787135, ssim=0.867549, lpips=0.092632
  image_022: loss=0.0006762045, psnr=31.699219, ssim=0.869518, lpips=0.096515
  image_023: loss=0.0006841407, psnr=31.648545, ssim=0.856434, lpips=0.120633
  image_024: loss=0.0009253815, psnr=30.336791, ssim=0.827117, lpips=0.127960
