====================================================================================================
saved_time: 2026-05-02 12:08:50
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0032547784_adam0p0003108585_decay362_trial014 --optimizer aux-sign-auto-cos-inc --lrate 0.0003108585461624725 --lrate_decay 362 --muon_lrate 0.0032547783766539683 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0032547784_adam0p0003108585_decay362_trial014/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0032547784_adam0p0003108585_decay362_trial014
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 31 min
current_train_loss: 0.0024226098
current_train_psnr: 33.263390
testset_mean_loss: 0.0012465617
testset_mean_psnr: 29.191724
testset_mean_ssim: 0.966251
testset_mean_lpips: 0.023547
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008737689, psnr=30.586034, ssim=0.970094, lpips=0.020425
  image_001: loss=0.0013347697, psnr=28.745936, ssim=0.965580, lpips=0.023745
  image_002: loss=0.0016031354, psnr=27.950298, ssim=0.960511, lpips=0.025350
  image_003: loss=0.0017097556, psnr=27.670659, ssim=0.957278, lpips=0.027423
  image_004: loss=0.0018542102, psnr=27.318410, ssim=0.957610, lpips=0.031054
  image_005: loss=0.0012841350, psnr=28.913893, ssim=0.965584, lpips=0.025810
  image_006: loss=0.0013167616, psnr=28.804928, ssim=0.965165, lpips=0.026092
  image_007: loss=0.0008797182, psnr=30.556564, ssim=0.974091, lpips=0.017419
  image_008: loss=0.0012697368, psnr=28.962863, ssim=0.961681, lpips=0.026257
  image_009: loss=0.0014584603, psnr=28.361053, ssim=0.958323, lpips=0.026311
  image_010: loss=0.0009421378, psnr=30.258855, ssim=0.973268, lpips=0.016816
  image_011: loss=0.0008189611, psnr=30.867367, ssim=0.978287, lpips=0.013622
  image_012: loss=0.0008323752, psnr=30.796808, ssim=0.976981, lpips=0.032354
  image_013: loss=0.0007759593, psnr=31.101610, ssim=0.977899, lpips=0.015263
  image_014: loss=0.0011942489, psnr=29.229051, ssim=0.969169, lpips=0.019994
  image_015: loss=0.0011105493, psnr=29.544621, ssim=0.965466, lpips=0.024065
  image_016: loss=0.0011679108, psnr=29.325903, ssim=0.972397, lpips=0.016481
  image_017: loss=0.0010854768, psnr=29.643794, ssim=0.971929, lpips=0.020638
  image_018: loss=0.0011578525, psnr=29.363467, ssim=0.968680, lpips=0.021663
  image_019: loss=0.0012829201, psnr=28.918003, ssim=0.966847, lpips=0.022635
  image_020: loss=0.0015161008, psnr=28.192719, ssim=0.960065, lpips=0.027856
  image_021: loss=0.0008772422, psnr=30.568804, ssim=0.968630, lpips=0.021870
  image_022: loss=0.0013865611, psnr=28.580610, ssim=0.958478, lpips=0.027131
  image_023: loss=0.0020964555, psnr=26.785143, ssim=0.952557, lpips=0.032863
  image_024: loss=0.0013348394, psnr=28.745710, ssim=0.959702, lpips=0.025551
