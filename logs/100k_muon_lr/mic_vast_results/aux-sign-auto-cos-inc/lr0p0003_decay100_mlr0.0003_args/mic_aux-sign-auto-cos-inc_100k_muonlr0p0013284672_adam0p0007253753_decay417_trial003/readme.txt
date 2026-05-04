====================================================================================================
saved_time: 2026-05-02 11:58:38
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/mic.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial003 --optimizer aux-sign-auto-cos-inc --lrate 0.0007253753172998836 --lrate_decay 417 --muon_lrate 0.001328467245876384 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial003/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial003
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 14 min
current_train_loss: 0.0013277216
current_train_psnr: 34.623260
testset_mean_loss: 0.0004733947
testset_mean_psnr: 33.447556
testset_mean_ssim: 0.979229
testset_mean_lpips: 0.023394
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004562323, psnr=33.408139, ssim=0.978612, lpips=0.015174
  image_001: loss=0.0002849867, psnr=35.451753, ssim=0.984094, lpips=0.019297
  image_002: loss=0.0005136327, psnr=32.893473, ssim=0.976683, lpips=0.026530
  image_003: loss=0.0004311506, psnr=33.653709, ssim=0.976707, lpips=0.027851
  image_004: loss=0.0004348130, psnr=33.616973, ssim=0.974263, lpips=0.028159
  image_005: loss=0.0003810882, psnr=34.189743, ssim=0.979352, lpips=0.026639
  image_006: loss=0.0003402718, psnr=34.681739, ssim=0.990229, lpips=0.013125
  image_007: loss=0.0004439534, psnr=33.526625, ssim=0.982068, lpips=0.015667
  image_008: loss=0.0005361600, psnr=32.707055, ssim=0.971148, lpips=0.037639
  image_009: loss=0.0004426494, psnr=33.539400, ssim=0.974921, lpips=0.022704
  image_010: loss=0.0002926216, psnr=35.336935, ssim=0.982320, lpips=0.016067
  image_011: loss=0.0002730990, psnr=35.636797, ssim=0.985559, lpips=0.017966
  image_012: loss=0.0011130348, psnr=29.534912, ssim=0.977267, lpips=0.022204
  image_013: loss=0.0006707370, psnr=31.734477, ssim=0.981898, lpips=0.015461
  image_014: loss=0.0005075916, psnr=32.944855, ssim=0.982151, lpips=0.016277
  image_015: loss=0.0004620996, psnr=33.352643, ssim=0.980496, lpips=0.018831
  image_016: loss=0.0004087583, psnr=33.885333, ssim=0.978279, lpips=0.019371
  image_017: loss=0.0004663534, psnr=33.312848, ssim=0.975671, lpips=0.028053
  image_018: loss=0.0003299368, psnr=34.815691, ssim=0.984720, lpips=0.015102
  image_019: loss=0.0005290900, psnr=32.764704, ssim=0.987812, lpips=0.013272
  image_020: loss=0.0005530038, psnr=32.572718, ssim=0.973178, lpips=0.039197
  image_021: loss=0.0005321708, psnr=32.739489, ssim=0.972202, lpips=0.041090
  image_022: loss=0.0005361079, psnr=32.707477, ssim=0.973777, lpips=0.033583
  image_023: loss=0.0005427311, psnr=32.654152, ssim=0.975636, lpips=0.031726
  image_024: loss=0.0003525934, psnr=34.527257, ssim=0.981690, lpips=0.023855
