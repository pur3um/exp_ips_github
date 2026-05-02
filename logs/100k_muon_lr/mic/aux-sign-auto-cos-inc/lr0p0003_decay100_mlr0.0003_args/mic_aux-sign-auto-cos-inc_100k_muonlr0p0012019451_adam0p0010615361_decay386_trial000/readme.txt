====================================================================================================
saved_time: 2026-04-29 17:41:23
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/mic.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0012019451_adam0p0010615361_decay386_trial000 --optimizer aux-sign-auto-cos-inc --lrate 0.0010615360772719654 --lrate_decay 386 --muon_lrate 0.0012019450985893186 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0012019451_adam0p0010615361_decay386_trial000/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0012019451_adam0p0010615361_decay386_trial000
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 5 min
current_train_loss: 0.0013118018
current_train_psnr: 34.242264
testset_mean_loss: 0.0004768144
testset_mean_psnr: 33.401491
testset_mean_ssim: 0.979062
testset_mean_lpips: 0.024262
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004458400, psnr=33.508209, ssim=0.978896, lpips=0.016577
  image_001: loss=0.0002914790, psnr=35.353926, ssim=0.983867, lpips=0.020965
  image_002: loss=0.0005387627, psnr=32.686024, ssim=0.976590, lpips=0.025827
  image_003: loss=0.0004438991, psnr=33.527157, ssim=0.976574, lpips=0.027512
  image_004: loss=0.0004404346, psnr=33.561184, ssim=0.973980, lpips=0.028431
  image_005: loss=0.0003832064, psnr=34.165671, ssim=0.979125, lpips=0.027411
  image_006: loss=0.0003542377, psnr=34.507052, ssim=0.989722, lpips=0.013504
  image_007: loss=0.0004382671, psnr=33.582610, ssim=0.982124, lpips=0.016272
  image_008: loss=0.0005556635, psnr=32.551881, ssim=0.970827, lpips=0.039744
  image_009: loss=0.0004484728, psnr=33.482638, ssim=0.974564, lpips=0.026190
  image_010: loss=0.0002981745, psnr=35.255293, ssim=0.982191, lpips=0.017854
  image_011: loss=0.0002829136, psnr=35.483461, ssim=0.985243, lpips=0.017931
  image_012: loss=0.0010992161, psnr=29.589169, ssim=0.978065, lpips=0.019653
  image_013: loss=0.0006154525, psnr=32.108054, ssim=0.982298, lpips=0.015315
  image_014: loss=0.0004946522, psnr=33.057000, ssim=0.982325, lpips=0.016178
  image_015: loss=0.0004567883, psnr=33.402849, ssim=0.980264, lpips=0.019359
  image_016: loss=0.0004089344, psnr=33.883463, ssim=0.977560, lpips=0.019488
  image_017: loss=0.0004982977, psnr=33.025110, ssim=0.976156, lpips=0.028166
  image_018: loss=0.0003406267, psnr=34.677212, ssim=0.984142, lpips=0.015927
  image_019: loss=0.0005225206, psnr=32.818965, ssim=0.987500, lpips=0.014016
  image_020: loss=0.0005724452, psnr=32.422660, ssim=0.972482, lpips=0.041425
  image_021: loss=0.0005423305, psnr=32.657359, ssim=0.971596, lpips=0.045283
  image_022: loss=0.0005367627, psnr=32.702176, ssim=0.973473, lpips=0.035045
  image_023: loss=0.0005515874, psnr=32.583856, ssim=0.975456, lpips=0.033570
  image_024: loss=0.0003593943, psnr=34.444287, ssim=0.981521, lpips=0.024897
