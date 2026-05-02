====================================================================================================
saved_time: 2026-04-30 23:56:27
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/lego.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/lego/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname lego_aux-sign-auto-cos-inc_100k_muonlr0p002664786_adam0p0029582881_decay230_trial010 --optimizer aux-sign-auto-cos-inc --lrate 0.002958288139689019 --lrate_decay 230 --muon_lrate 0.0026647860197380052 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/lego/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/lego_aux-sign-auto-cos-inc_100k_muonlr0p002664786_adam0p0029582881_decay230_trial010/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: lego_aux-sign-auto-cos-inc_100k_muonlr0p002664786_adam0p0029582881_decay230_trial010
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 8 min
current_train_loss: 0.0026485214
current_train_psnr: 34.316933
testset_mean_loss: 0.0006609572
testset_mean_psnr: 32.059759
testset_mean_ssim: 0.968687
testset_mean_lpips: 0.017800
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004699126, psnr=33.279828, ssim=0.972225, lpips=0.013787
  image_001: loss=0.0012770097, psnr=28.938058, ssim=0.959303, lpips=0.025005
  image_002: loss=0.0003889495, psnr=34.101067, ssim=0.976101, lpips=0.010976
  image_003: loss=0.0004695583, psnr=33.283104, ssim=0.973331, lpips=0.016654
  image_004: loss=0.0006860273, psnr=31.636585, ssim=0.967025, lpips=0.018349
  image_005: loss=0.0004344493, psnr=33.620607, ssim=0.972097, lpips=0.019653
  image_006: loss=0.0006482113, psnr=31.882833, ssim=0.969759, lpips=0.018846
  image_007: loss=0.0004266445, psnr=33.699338, ssim=0.969856, lpips=0.014912
  image_008: loss=0.0005138916, psnr=32.891284, ssim=0.973916, lpips=0.013712
  image_009: loss=0.0009657795, psnr=30.151220, ssim=0.969913, lpips=0.021635
  image_010: loss=0.0013761792, psnr=28.613250, ssim=0.956345, lpips=0.029873
  image_011: loss=0.0010524696, psnr=29.777904, ssim=0.959022, lpips=0.023086
  image_012: loss=0.0006817664, psnr=31.663644, ssim=0.968676, lpips=0.016787
  image_013: loss=0.0006254099, psnr=32.038351, ssim=0.968657, lpips=0.018550
  image_014: loss=0.0005197474, psnr=32.842076, ssim=0.968920, lpips=0.016479
  image_015: loss=0.0006027161, psnr=32.198872, ssim=0.970017, lpips=0.020232
  image_016: loss=0.0006603345, psnr=31.802359, ssim=0.976832, lpips=0.017127
  image_017: loss=0.0006257801, psnr=32.035782, ssim=0.969904, lpips=0.019592
  image_018: loss=0.0005671854, psnr=32.462749, ssim=0.970152, lpips=0.015152
  image_019: loss=0.0005214355, psnr=32.827993, ssim=0.965427, lpips=0.017688
  image_020: loss=0.0003543934, psnr=34.505143, ssim=0.972365, lpips=0.012107
  image_021: loss=0.0005422192, psnr=32.658250, ssim=0.969967, lpips=0.012991
  image_022: loss=0.0007397651, psnr=31.309061, ssim=0.965823, lpips=0.020698
  image_023: loss=0.0007265355, psnr=31.387431, ssim=0.966060, lpips=0.016416
  image_024: loss=0.0006475604, psnr=31.887196, ssim=0.965473, lpips=0.014682
