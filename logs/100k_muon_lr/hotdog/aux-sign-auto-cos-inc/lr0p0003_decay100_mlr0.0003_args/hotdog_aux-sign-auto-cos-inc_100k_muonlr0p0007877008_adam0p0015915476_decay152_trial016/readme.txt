====================================================================================================
saved_time: 2026-05-03 01:28:11
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0007877008_adam0p0015915476_decay152_trial016 --optimizer aux-sign-auto-cos-inc --lrate 0.0015915476453185764 --lrate_decay 152 --muon_lrate 0.0007877008324421214 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0007877008_adam0p0015915476_decay152_trial016/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0007877008_adam0p0015915476_decay152_trial016
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 34 min
current_train_loss: 0.0011120740
current_train_psnr: 34.937653
testset_mean_loss: 0.0002610788
testset_mean_psnr: 36.586684
testset_mean_ssim: 0.978635
testset_mean_lpips: 0.020646
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0002227238, psnr=36.522332, ssim=0.979733, lpips=0.019394
  image_001: loss=0.0001992647, psnr=37.005695, ssim=0.981880, lpips=0.020721
  image_002: loss=0.0002639850, psnr=35.784206, ssim=0.976716, lpips=0.030404
  image_003: loss=0.0001729843, psnr=37.619932, ssim=0.980250, lpips=0.022500
  image_004: loss=0.0001302951, psnr=38.850717, ssim=0.983266, lpips=0.016668
  image_005: loss=0.0001766270, psnr=37.529428, ssim=0.982136, lpips=0.015463
  image_006: loss=0.0002384107, psnr=36.226741, ssim=0.980748, lpips=0.015633
  image_007: loss=0.0001575769, psnr=38.025073, ssim=0.983619, lpips=0.014615
  image_008: loss=0.0001673450, psnr=37.763869, ssim=0.977769, lpips=0.016705
  image_009: loss=0.0001956701, psnr=37.084753, ssim=0.971043, lpips=0.027851
  image_010: loss=0.0001682665, psnr=37.740022, ssim=0.973186, lpips=0.027582
  image_011: loss=0.0002888840, psnr=35.392764, ssim=0.972058, lpips=0.028390
  image_012: loss=0.0002867552, psnr=35.424885, ssim=0.982176, lpips=0.013716
  image_013: loss=0.0001906223, psnr=37.198260, ssim=0.987112, lpips=0.011478
  image_014: loss=0.0003218427, psnr=34.923562, ssim=0.981699, lpips=0.018739
  image_015: loss=0.0013544213, psnr=28.682462, ssim=0.962250, lpips=0.031638
  image_016: loss=0.0005430584, psnr=32.651534, ssim=0.967194, lpips=0.030149
  image_017: loss=0.0001790539, psnr=37.470160, ssim=0.980913, lpips=0.017917
  image_018: loss=0.0002562179, psnr=35.913904, ssim=0.978125, lpips=0.015542
  image_019: loss=0.0002021264, psnr=36.943767, ssim=0.982081, lpips=0.014242
  image_020: loss=0.0001333141, psnr=38.751236, ssim=0.984632, lpips=0.014132
  image_021: loss=0.0001382718, psnr=38.592661, ssim=0.980645, lpips=0.017930
  image_022: loss=0.0001475425, psnr=38.310826, ssim=0.978570, lpips=0.027656
  image_023: loss=0.0001667463, psnr=37.779436, ssim=0.978947, lpips=0.026313
  image_024: loss=0.0002249640, psnr=36.478868, ssim=0.979126, lpips=0.020766
