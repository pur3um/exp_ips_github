====================================================================================================
saved_time: 2026-05-01 21:19:50
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/lego.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/lego/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname lego_aux-sign-auto-cos-inc_100k_muonlr0p0032797262_adam0p0005110569_decay386_trial020 --optimizer aux-sign-auto-cos-inc --lrate 0.000511056882002473 --lrate_decay 386 --muon_lrate 0.003279726192347939 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/lego/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/lego_aux-sign-auto-cos-inc_100k_muonlr0p0032797262_adam0p0005110569_decay386_trial020/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: lego_aux-sign-auto-cos-inc_100k_muonlr0p0032797262_adam0p0005110569_decay386_trial020
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 8 min
current_train_loss: 0.0025803600
current_train_psnr: 34.384300
testset_mean_loss: 0.0006672746
testset_mean_psnr: 32.006899
testset_mean_ssim: 0.968564
testset_mean_lpips: 0.017652
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004629229, psnr=33.344913, ssim=0.972574, lpips=0.013870
  image_001: loss=0.0011917653, psnr=29.238092, ssim=0.962543, lpips=0.022278
  image_002: loss=0.0003877913, psnr=34.114018, ssim=0.975846, lpips=0.010806
  image_003: loss=0.0004771420, psnr=33.213522, ssim=0.973363, lpips=0.016646
  image_004: loss=0.0006984579, psnr=31.558597, ssim=0.966909, lpips=0.017776
  image_005: loss=0.0004487534, psnr=33.479921, ssim=0.972118, lpips=0.015071
  image_006: loss=0.0006897532, psnr=31.613062, ssim=0.969062, lpips=0.019025
  image_007: loss=0.0004292048, psnr=33.673354, ssim=0.969464, lpips=0.015058
  image_008: loss=0.0005119884, psnr=32.907398, ssim=0.973276, lpips=0.014283
  image_009: loss=0.0009947680, psnr=30.022782, ssim=0.969300, lpips=0.022095
  image_010: loss=0.0013880123, psnr=28.576066, ssim=0.956217, lpips=0.031628
  image_011: loss=0.0010548468, psnr=29.768106, ssim=0.959380, lpips=0.023532
  image_012: loss=0.0007013906, psnr=31.540400, ssim=0.968675, lpips=0.016723
  image_013: loss=0.0006465025, psnr=31.894297, ssim=0.968261, lpips=0.018271
  image_014: loss=0.0005261075, psnr=32.789254, ssim=0.968663, lpips=0.015398
  image_015: loss=0.0006112023, psnr=32.138150, ssim=0.969447, lpips=0.020852
  image_016: loss=0.0006459111, psnr=31.898272, ssim=0.976904, lpips=0.017290
  image_017: loss=0.0006394483, psnr=31.941945, ssim=0.969496, lpips=0.020099
  image_018: loss=0.0006046992, psnr=32.184605, ssim=0.968086, lpips=0.016113
  image_019: loss=0.0005159122, psnr=32.874241, ssim=0.965138, lpips=0.017761
  image_020: loss=0.0003650737, psnr=34.376193, ssim=0.972131, lpips=0.013020
  image_021: loss=0.0005512355, psnr=32.586627, ssim=0.969580, lpips=0.012263
  image_022: loss=0.0007569335, psnr=31.209422, ssim=0.965933, lpips=0.020452
  image_023: loss=0.0007367621, psnr=31.326726, ssim=0.965753, lpips=0.017330
  image_024: loss=0.0006452815, psnr=31.902507, ssim=0.965975, lpips=0.013647
