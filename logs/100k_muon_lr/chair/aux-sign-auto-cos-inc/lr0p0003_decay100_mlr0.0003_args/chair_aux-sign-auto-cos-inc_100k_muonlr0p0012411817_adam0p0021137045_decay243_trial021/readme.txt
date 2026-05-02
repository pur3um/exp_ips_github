====================================================================================================
saved_time: 2026-05-01 23:16:39
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/chair.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/chair/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname chair_aux-sign-auto-cos-inc_100k_muonlr0p0012411817_adam0p0021137045_decay243_trial021 --optimizer aux-sign-auto-cos-inc --lrate 0.0021137045139821564 --lrate_decay 243 --muon_lrate 0.0012411817260249047 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/chair/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/chair_aux-sign-auto-cos-inc_100k_muonlr0p0012411817_adam0p0021137045_decay243_trial021/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: chair_aux-sign-auto-cos-inc_100k_muonlr0p0012411817_adam0p0021137045_decay243_trial021
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 6 min
current_train_loss: 0.0018337514
current_train_psnr: 32.222603
testset_mean_loss: 0.0004070077
testset_mean_psnr: 34.129040
testset_mean_ssim: 0.977505
testset_mean_lpips: 0.016842
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001451707, psnr=38.381206, ssim=0.992981, lpips=0.006376
  image_001: loss=0.0001988962, psnr=37.013734, ssim=0.988328, lpips=0.006510
  image_002: loss=0.0004122965, psnr=33.847903, ssim=0.979399, lpips=0.017131
  image_003: loss=0.0004287923, psnr=33.677529, ssim=0.977748, lpips=0.016624
  image_004: loss=0.0004828115, psnr=33.162223, ssim=0.975750, lpips=0.017591
  image_005: loss=0.0005173804, psnr=32.861899, ssim=0.973954, lpips=0.016507
  image_006: loss=0.0005301015, psnr=32.756409, ssim=0.970887, lpips=0.019097
  image_007: loss=0.0005564337, psnr=32.545865, ssim=0.967525, lpips=0.024955
  image_008: loss=0.0004082035, psnr=33.891232, ssim=0.974268, lpips=0.015980
  image_009: loss=0.0004535836, psnr=33.433425, ssim=0.973292, lpips=0.018912
  image_010: loss=0.0006097990, psnr=32.148132, ssim=0.971258, lpips=0.027510
  image_011: loss=0.0003940103, psnr=34.044923, ssim=0.978128, lpips=0.019025
  image_012: loss=0.0002874466, psnr=35.414427, ssim=0.982360, lpips=0.016558
  image_013: loss=0.0003570127, psnr=34.473162, ssim=0.979110, lpips=0.021105
  image_014: loss=0.0002804703, psnr=35.521130, ssim=0.982883, lpips=0.015076
  image_015: loss=0.0004245720, psnr=33.720485, ssim=0.978232, lpips=0.020832
  image_016: loss=0.0003454851, psnr=34.615705, ssim=0.981363, lpips=0.014194
  image_017: loss=0.0004931056, psnr=33.070600, ssim=0.975860, lpips=0.017066
  image_018: loss=0.0005341405, psnr=32.723444, ssim=0.973328, lpips=0.017457
  image_019: loss=0.0005363881, psnr=32.705208, ssim=0.969252, lpips=0.019143
  image_020: loss=0.0004553531, psnr=33.416516, ssim=0.971927, lpips=0.018283
  image_021: loss=0.0003719724, psnr=34.294891, ssim=0.975506, lpips=0.015832
  image_022: loss=0.0004066338, psnr=33.907964, ssim=0.974625, lpips=0.018728
  image_023: loss=0.0003438241, psnr=34.636636, ssim=0.980501, lpips=0.013804
  image_024: loss=0.0002013098, psnr=36.961348, ssim=0.989159, lpips=0.006764
