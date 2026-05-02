====================================================================================================
saved_time: 2026-05-01 17:32:19
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0003547221_adam0p0003346566_decay138_trial007 --optimizer aux-sign-auto-cos-inc --lrate 0.0003346565669014132 --lrate_decay 138 --muon_lrate 0.00035472207152279926 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0003547221_adam0p0003346566_decay138_trial007/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0003547221_adam0p0003346566_decay138_trial007
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0014631670
current_train_psnr: 33.964363
testset_mean_loss: 0.0003503583
testset_mean_psnr: 35.010642
testset_mean_ssim: 0.971933
testset_mean_lpips: 0.031440
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0003370025, psnr=34.723667, ssim=0.971536, lpips=0.030342
  image_001: loss=0.0003173565, psnr=34.984525, ssim=0.972864, lpips=0.034254
  image_002: loss=0.0003773226, psnr=34.232871, ssim=0.969635, lpips=0.035407
  image_003: loss=0.0002713063, psnr=35.665399, ssim=0.972317, lpips=0.033842
  image_004: loss=0.0002009109, psnr=36.969963, ssim=0.975797, lpips=0.029352
  image_005: loss=0.0002696167, psnr=35.692530, ssim=0.975456, lpips=0.026289
  image_006: loss=0.0003413107, psnr=34.668500, ssim=0.973268, lpips=0.026163
  image_007: loss=0.0002406673, psnr=36.185828, ssim=0.977193, lpips=0.024018
  image_008: loss=0.0002393444, psnr=36.209765, ssim=0.970850, lpips=0.027208
  image_009: loss=0.0002546162, psnr=35.941138, ssim=0.963550, lpips=0.041159
  image_010: loss=0.0002361685, psnr=36.267778, ssim=0.964683, lpips=0.042072
  image_011: loss=0.0004165023, psnr=33.803825, ssim=0.965812, lpips=0.039690
  image_012: loss=0.0003951782, psnr=34.032069, ssim=0.977273, lpips=0.021373
  image_013: loss=0.0003490145, psnr=34.571564, ssim=0.980361, lpips=0.019554
  image_014: loss=0.0004424156, psnr=33.541695, ssim=0.975849, lpips=0.028951
  image_015: loss=0.0013563473, psnr=28.676290, ssim=0.961003, lpips=0.037561
  image_016: loss=0.0005901315, psnr=32.290511, ssim=0.964415, lpips=0.035895
  image_017: loss=0.0002627368, psnr=35.804790, ssim=0.974466, lpips=0.027013
  image_018: loss=0.0003608900, psnr=34.426250, ssim=0.971625, lpips=0.023815
  image_019: loss=0.0003003896, psnr=35.223150, ssim=0.974771, lpips=0.025729
  image_020: loss=0.0002011274, psnr=36.965286, ssim=0.978531, lpips=0.024956
  image_021: loss=0.0002067206, psnr=36.846160, ssim=0.973485, lpips=0.030362
  image_022: loss=0.0002180603, psnr=36.614232, ssim=0.970459, lpips=0.042281
  image_023: loss=0.0002473845, psnr=36.066274, ssim=0.971209, lpips=0.040978
  image_024: loss=0.0003264372, psnr=34.862002, ssim=0.971924, lpips=0.037743
