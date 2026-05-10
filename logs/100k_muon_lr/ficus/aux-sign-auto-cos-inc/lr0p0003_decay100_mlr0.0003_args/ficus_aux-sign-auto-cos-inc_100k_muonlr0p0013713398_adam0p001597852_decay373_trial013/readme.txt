====================================================================================================
saved_time: 2026-05-02 08:36:51
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0013713398_adam0p001597852_decay373_trial013 --optimizer aux-sign-auto-cos-inc --lrate 0.0015978519905430666 --lrate_decay 373 --muon_lrate 0.0013713398204094074 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0013713398_adam0p001597852_decay373_trial013/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0013713398_adam0p001597852_decay373_trial013
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 26 min
current_train_loss: 0.0020057133
current_train_psnr: 33.821388
testset_mean_loss: 0.0012522560
testset_mean_psnr: 29.183670
testset_mean_ssim: 0.966022
testset_mean_lpips: 0.025286
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008776484, psnr=30.566794, ssim=0.970350, lpips=0.021220
  image_001: loss=0.0013775153, psnr=28.609035, ssim=0.964585, lpips=0.027042
  image_002: loss=0.0015541587, psnr=28.085046, ssim=0.961500, lpips=0.027764
  image_003: loss=0.0017568849, psnr=27.552567, ssim=0.956599, lpips=0.030253
  image_004: loss=0.0018667202, psnr=27.289208, ssim=0.956184, lpips=0.035379
  image_005: loss=0.0012626898, psnr=28.987033, ssim=0.965886, lpips=0.025448
  image_006: loss=0.0012970649, psnr=28.870382, ssim=0.965974, lpips=0.025411
  image_007: loss=0.0008593520, psnr=30.658289, ssim=0.973991, lpips=0.018397
  image_008: loss=0.0013120469, psnr=28.820506, ssim=0.961575, lpips=0.028914
  image_009: loss=0.0015215771, psnr=28.177060, ssim=0.956572, lpips=0.030216
  image_010: loss=0.0009221011, psnr=30.352214, ssim=0.973532, lpips=0.017983
  image_011: loss=0.0008279665, psnr=30.819872, ssim=0.977972, lpips=0.014634
  image_012: loss=0.0008418919, psnr=30.747436, ssim=0.976360, lpips=0.036393
  image_013: loss=0.0007538293, psnr=31.227269, ssim=0.978300, lpips=0.015626
  image_014: loss=0.0011620094, psnr=29.347903, ssim=0.969054, lpips=0.020592
  image_015: loss=0.0011132814, psnr=29.533950, ssim=0.964973, lpips=0.023449
  image_016: loss=0.0011789398, psnr=29.285083, ssim=0.972622, lpips=0.018874
  image_017: loss=0.0011106409, psnr=29.544263, ssim=0.971753, lpips=0.021324
  image_018: loss=0.0011647145, psnr=29.337805, ssim=0.968417, lpips=0.020804
  image_019: loss=0.0013238048, psnr=28.781760, ssim=0.965381, lpips=0.023252
  image_020: loss=0.0014833327, psnr=28.287614, ssim=0.960660, lpips=0.029888
  image_021: loss=0.0008081431, psnr=30.925117, ssim=0.969709, lpips=0.024323
  image_022: loss=0.0013664555, psnr=28.644045, ssim=0.959079, lpips=0.029725
  image_023: loss=0.0021193905, psnr=26.737890, ssim=0.950974, lpips=0.036557
  image_024: loss=0.0014442407, psnr=28.403604, ssim=0.958545, lpips=0.028672
