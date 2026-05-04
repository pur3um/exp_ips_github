====================================================================================================
saved_time: 2026-05-03 21:33:15
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/mic.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0036040037_adam0p0018872487_decay143_trial018 --optimizer aux-sign-auto-cos-inc --lrate 0.0018872487361256125 --lrate_decay 143 --muon_lrate 0.003604003749222079 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0036040037_adam0p0018872487_decay143_trial018/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0036040037_adam0p0018872487_decay143_trial018
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0016563934
current_train_psnr: 33.043999
testset_mean_loss: 0.0004560351
testset_mean_psnr: 33.578084
testset_mean_ssim: 0.979988
testset_mean_lpips: 0.021992
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004286522, psnr=33.678948, ssim=0.980058, lpips=0.014177
  image_001: loss=0.0002909067, psnr=35.362461, ssim=0.984335, lpips=0.020192
  image_002: loss=0.0004882527, psnr=33.113553, ssim=0.978652, lpips=0.024776
  image_003: loss=0.0004123057, psnr=33.847806, ssim=0.978048, lpips=0.025671
  image_004: loss=0.0004197168, psnr=33.770436, ssim=0.975072, lpips=0.027607
  image_005: loss=0.0003770329, psnr=34.236206, ssim=0.979454, lpips=0.027380
  image_006: loss=0.0003270612, psnr=34.853709, ssim=0.990941, lpips=0.010919
  image_007: loss=0.0004119483, psnr=33.851572, ssim=0.983451, lpips=0.012339
  image_008: loss=0.0005351449, psnr=32.715285, ssim=0.971155, lpips=0.036370
  image_009: loss=0.0004315832, psnr=33.649353, ssim=0.975703, lpips=0.021196
  image_010: loss=0.0003006296, psnr=35.219681, ssim=0.982687, lpips=0.015397
  image_011: loss=0.0002702660, psnr=35.682084, ssim=0.985602, lpips=0.016969
  image_012: loss=0.0009348819, psnr=30.292432, ssim=0.978418, lpips=0.020619
  image_013: loss=0.0006979308, psnr=31.561876, ssim=0.981394, lpips=0.016106
  image_014: loss=0.0005243957, psnr=32.803408, ssim=0.982890, lpips=0.016153
  image_015: loss=0.0004520820, psnr=33.447827, ssim=0.981617, lpips=0.018474
  image_016: loss=0.0004000048, psnr=33.979347, ssim=0.978558, lpips=0.019147
  image_017: loss=0.0004515170, psnr=33.453258, ssim=0.976021, lpips=0.026731
  image_018: loss=0.0003185713, psnr=34.967932, ssim=0.984901, lpips=0.012711
  image_019: loss=0.0004766066, psnr=33.218399, ssim=0.989242, lpips=0.011196
  image_020: loss=0.0005233283, psnr=32.812257, ssim=0.974520, lpips=0.035348
  image_021: loss=0.0005194996, psnr=32.844147, ssim=0.973215, lpips=0.040577
  image_022: loss=0.0005313958, psnr=32.745818, ssim=0.974721, lpips=0.029563
  image_023: loss=0.0005282230, psnr=32.771826, ssim=0.976710, lpips=0.028730
  image_024: loss=0.0003489414, psnr=34.572473, ssim=0.982347, lpips=0.021460
