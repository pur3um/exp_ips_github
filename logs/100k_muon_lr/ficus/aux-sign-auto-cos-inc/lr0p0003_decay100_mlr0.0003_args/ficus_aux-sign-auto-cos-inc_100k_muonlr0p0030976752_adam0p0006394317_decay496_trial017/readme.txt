====================================================================================================
saved_time: 2026-05-02 22:49:01
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0030976752_adam0p0006394317_decay496_trial017 --optimizer aux-sign-auto-cos-inc --lrate 0.0006394317431824077 --lrate_decay 496 --muon_lrate 0.003097675170160024 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0030976752_adam0p0006394317_decay496_trial017/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0030976752_adam0p0006394317_decay496_trial017
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 33 min
current_train_loss: 0.0023425713
current_train_psnr: 32.323975
testset_mean_loss: 0.0012549161
testset_mean_psnr: 29.161605
testset_mean_ssim: 0.966121
testset_mean_lpips: 0.023289
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008586324, psnr=30.661927, ssim=0.970628, lpips=0.021236
  image_001: loss=0.0013777646, psnr=28.608249, ssim=0.964333, lpips=0.024518
  image_002: loss=0.0016356196, psnr=27.863177, ssim=0.959459, lpips=0.027343
  image_003: loss=0.0017800721, psnr=27.495624, ssim=0.955211, lpips=0.029687
  image_004: loss=0.0019108171, psnr=27.187809, ssim=0.956213, lpips=0.031477
  image_005: loss=0.0012841547, psnr=28.913826, ssim=0.965568, lpips=0.025792
  image_006: loss=0.0013183312, psnr=28.799754, ssim=0.965900, lpips=0.024636
  image_007: loss=0.0008870685, psnr=30.520428, ssim=0.973897, lpips=0.016841
  image_008: loss=0.0013002330, psnr=28.859788, ssim=0.961482, lpips=0.026405
  image_009: loss=0.0015176291, psnr=28.188343, ssim=0.957613, lpips=0.028541
  image_010: loss=0.0009546716, psnr=30.201459, ssim=0.973410, lpips=0.017459
  image_011: loss=0.0008218599, psnr=30.852021, ssim=0.978378, lpips=0.014472
  image_012: loss=0.0008239324, psnr=30.841084, ssim=0.977264, lpips=0.018545
  image_013: loss=0.0007958718, psnr=30.991568, ssim=0.977716, lpips=0.014548
  image_014: loss=0.0011824426, psnr=29.272199, ssim=0.968908, lpips=0.021673
  image_015: loss=0.0011184131, psnr=29.513977, ssim=0.966011, lpips=0.023885
  image_016: loss=0.0011641133, psnr=29.340047, ssim=0.972519, lpips=0.017192
  image_017: loss=0.0011005648, psnr=29.583843, ssim=0.972148, lpips=0.020659
  image_018: loss=0.0012003527, psnr=29.206911, ssim=0.968977, lpips=0.021326
  image_019: loss=0.0013005475, psnr=28.858737, ssim=0.966516, lpips=0.022023
  image_020: loss=0.0015034168, psnr=28.229206, ssim=0.959845, lpips=0.027083
  image_021: loss=0.0008646483, psnr=30.631604, ssim=0.968387, lpips=0.021857
  image_022: loss=0.0013673595, psnr=28.641172, ssim=0.959222, lpips=0.028170
  image_023: loss=0.0019449964, psnr=27.110812, ssim=0.953577, lpips=0.032523
  image_024: loss=0.0013593900, psnr=28.666559, ssim=0.959843, lpips=0.024335
