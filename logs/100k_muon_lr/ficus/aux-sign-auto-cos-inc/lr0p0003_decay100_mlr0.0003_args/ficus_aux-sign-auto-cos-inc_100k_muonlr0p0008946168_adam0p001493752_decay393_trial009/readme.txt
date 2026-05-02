====================================================================================================
saved_time: 2026-05-01 18:46:48
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0008946168_adam0p001493752_decay393_trial009 --optimizer aux-sign-auto-cos-inc --lrate 0.0014937519612696593 --lrate_decay 393 --muon_lrate 0.0008946167893901963 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0008946168_adam0p001493752_decay393_trial009/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0008946168_adam0p001493752_decay393_trial009
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 26 min
current_train_loss: 0.0025231801
current_train_psnr: 31.970524
testset_mean_loss: 0.0013410285
testset_mean_psnr: 28.887579
testset_mean_ssim: 0.963880
testset_mean_lpips: 0.026786
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0009113566, psnr=30.403116, ssim=0.969043, lpips=0.023868
  image_001: loss=0.0014239213, psnr=28.465140, ssim=0.962383, lpips=0.028236
  image_002: loss=0.0016894263, psnr=27.722607, ssim=0.957982, lpips=0.030590
  image_003: loss=0.0017671421, psnr=27.527285, ssim=0.954219, lpips=0.032564
  image_004: loss=0.0019104442, psnr=27.188656, ssim=0.954622, lpips=0.034758
  image_005: loss=0.0013347208, psnr=28.746096, ssim=0.964091, lpips=0.029810
  image_006: loss=0.0014372064, psnr=28.424808, ssim=0.963917, lpips=0.028715
  image_007: loss=0.0009312221, psnr=30.309467, ssim=0.971930, lpips=0.020926
  image_008: loss=0.0015295113, psnr=28.154473, ssim=0.957785, lpips=0.031802
  image_009: loss=0.0016099621, psnr=27.931843, ssim=0.954285, lpips=0.029877
  image_010: loss=0.0009962146, psnr=30.016471, ssim=0.970881, lpips=0.018893
  image_011: loss=0.0008866696, psnr=30.522381, ssim=0.976284, lpips=0.017000
  image_012: loss=0.0008970472, psnr=30.471847, ssim=0.975141, lpips=0.017568
  image_013: loss=0.0007960298, psnr=30.990706, ssim=0.977189, lpips=0.017026
  image_014: loss=0.0012341269, psnr=29.086402, ssim=0.966536, lpips=0.023193
  image_015: loss=0.0011427274, psnr=29.420573, ssim=0.963773, lpips=0.026400
  image_016: loss=0.0012256533, psnr=29.116323, ssim=0.971098, lpips=0.020840
  image_017: loss=0.0011373934, psnr=29.440893, ssim=0.970172, lpips=0.023677
  image_018: loss=0.0012744532, psnr=28.946761, ssim=0.966634, lpips=0.023909
  image_019: loss=0.0013990725, psnr=28.541598, ssim=0.964137, lpips=0.025777
  image_020: loss=0.0016653818, psnr=27.784862, ssim=0.957336, lpips=0.034708
  image_021: loss=0.0009133014, psnr=30.393858, ssim=0.966768, lpips=0.027029
  image_022: loss=0.0015252464, psnr=28.166600, ssim=0.955347, lpips=0.033537
  image_023: loss=0.0023453448, psnr=26.297933, ssim=0.949199, lpips=0.039978
  image_024: loss=0.0015421378, psnr=28.118768, ssim=0.956256, lpips=0.028964
