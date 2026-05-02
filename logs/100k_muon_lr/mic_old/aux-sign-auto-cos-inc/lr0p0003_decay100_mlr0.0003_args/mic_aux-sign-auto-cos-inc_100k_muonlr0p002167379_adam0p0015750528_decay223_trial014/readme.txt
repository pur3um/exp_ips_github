====================================================================================================
saved_time: 2026-05-02 02:38:03
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p002167379_adam0p0015750528_decay223_trial014 --optimizer aux-sign-auto-cos-inc --lrate 0.0015750527540490304 --lrate_decay 223 --muon_lrate 0.0021673790355043686 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p002167379_adam0p0015750528_decay223_trial014/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p002167379_adam0p0015750528_decay223_trial014
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0012802137
current_train_psnr: 37.007423
testset_mean_loss: 0.0004461283
testset_mean_psnr: 33.706346
testset_mean_ssim: 0.980505
testset_mean_lpips: 0.021492
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004242024, psnr=33.724268, ssim=0.980184, lpips=0.014505
  image_001: loss=0.0002717326, psnr=35.658581, ssim=0.984942, lpips=0.017831
  image_002: loss=0.0004950811, psnr=33.053235, ssim=0.978689, lpips=0.023846
  image_003: loss=0.0004114585, psnr=33.856738, ssim=0.978064, lpips=0.024591
  image_004: loss=0.0004262473, psnr=33.703383, ssim=0.975666, lpips=0.026349
  image_005: loss=0.0003599801, psnr=34.437214, ssim=0.980277, lpips=0.027817
  image_006: loss=0.0003105623, psnr=35.078511, ssim=0.991295, lpips=0.011507
  image_007: loss=0.0003893958, psnr=34.096087, ssim=0.983842, lpips=0.013259
  image_008: loss=0.0005244035, psnr=32.803343, ssim=0.971405, lpips=0.035006
  image_009: loss=0.0004378036, psnr=33.587205, ssim=0.975862, lpips=0.021860
  image_010: loss=0.0002838467, psnr=35.469160, ssim=0.983265, lpips=0.014704
  image_011: loss=0.0002568772, psnr=35.902742, ssim=0.986410, lpips=0.016367
  image_012: loss=0.0010720548, psnr=29.697830, ssim=0.978352, lpips=0.018831
  image_013: loss=0.0005706643, psnr=32.436192, ssim=0.984200, lpips=0.013330
  image_014: loss=0.0005115184, psnr=32.911387, ssim=0.983038, lpips=0.015554
  image_015: loss=0.0004350650, psnr=33.614458, ssim=0.981987, lpips=0.017983
  image_016: loss=0.0003809402, psnr=34.191431, ssim=0.979368, lpips=0.018073
  image_017: loss=0.0004240261, psnr=33.726073, ssim=0.976859, lpips=0.026058
  image_018: loss=0.0003072429, psnr=35.125179, ssim=0.985394, lpips=0.013006
  image_019: loss=0.0004669971, psnr=33.306857, ssim=0.989741, lpips=0.011691
  image_020: loss=0.0005194228, psnr=32.844789, ssim=0.974493, lpips=0.036739
  image_021: loss=0.0005088717, psnr=32.933916, ssim=0.973819, lpips=0.039155
  image_022: loss=0.0005189102, psnr=32.849077, ssim=0.974882, lpips=0.029705
  image_023: loss=0.0005074270, psnr=32.946263, ssim=0.977581, lpips=0.028209
  image_024: loss=0.0003384755, psnr=34.704727, ssim=0.982999, lpips=0.021319
