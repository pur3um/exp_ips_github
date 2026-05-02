====================================================================================================
saved_time: 2026-04-30 22:35:50
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial006 --optimizer aux-sign-auto-cos-inc --lrate 0.0011095983846308643 --lrate_decay 471 --muon_lrate 0.0003663671900803624 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial006/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0003663672_adam0p0011095984_decay471_trial006
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0019232153
current_train_psnr: 33.627808
testset_mean_loss: 0.0006105944
testset_mean_psnr: 32.245049
testset_mean_ssim: 0.973124
testset_mean_lpips: 0.035692
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0005643415, psnr=32.484579, ssim=0.972917, lpips=0.023402
  image_001: loss=0.0003983401, psnr=33.997459, ssim=0.976873, lpips=0.032322
  image_002: loss=0.0006939813, psnr=31.586522, ssim=0.970166, lpips=0.037989
  image_003: loss=0.0005785672, psnr=32.376461, ssim=0.970357, lpips=0.040676
  image_004: loss=0.0005418921, psnr=32.660871, ssim=0.969115, lpips=0.043031
  image_005: loss=0.0005198844, psnr=32.840931, ssim=0.971651, lpips=0.042251
  image_006: loss=0.0006061962, psnr=32.173867, ssim=0.979002, lpips=0.027116
  image_007: loss=0.0006391550, psnr=31.943937, ssim=0.972540, lpips=0.031663
  image_008: loss=0.0006367696, psnr=31.960176, ssim=0.966563, lpips=0.051222
  image_009: loss=0.0005614226, psnr=32.507100, ssim=0.971212, lpips=0.034298
  image_010: loss=0.0003933793, psnr=34.051883, ssim=0.977843, lpips=0.026915
  image_011: loss=0.0004458383, psnr=33.508225, ssim=0.980077, lpips=0.027086
  image_012: loss=0.0011140679, psnr=29.530883, ssim=0.975918, lpips=0.022195
  image_013: loss=0.0007461911, psnr=31.271499, ssim=0.980706, lpips=0.017376
  image_014: loss=0.0006493334, psnr=31.875322, ssim=0.978148, lpips=0.022739
  image_015: loss=0.0005896538, psnr=32.294028, ssim=0.975893, lpips=0.026868
  image_016: loss=0.0005631550, psnr=32.493720, ssim=0.973209, lpips=0.025634
  image_017: loss=0.0005884306, psnr=32.303047, ssim=0.971390, lpips=0.034820
  image_018: loss=0.0004746043, psnr=33.236682, ssim=0.976430, lpips=0.029122
  image_019: loss=0.0007820942, psnr=31.067409, ssim=0.977897, lpips=0.026019
  image_020: loss=0.0006604075, psnr=31.801879, ssim=0.966152, lpips=0.061337
  image_021: loss=0.0006424979, psnr=31.921282, ssim=0.966083, lpips=0.059613
  image_022: loss=0.0006388349, psnr=31.946113, ssim=0.967509, lpips=0.055000
  image_023: loss=0.0007160132, psnr=31.450789, ssim=0.967223, lpips=0.052570
  image_024: loss=0.0005198079, psnr=32.841570, ssim=0.973215, lpips=0.041041
