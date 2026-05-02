====================================================================================================
saved_time: 2026-04-30 18:49:55
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial002 --optimizer aux-sign-auto-cos-inc --lrate 0.0008216908785704095 --lrate_decay 457 --muon_lrate 0.004514102265895567 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial002/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p0045141023_adam0p0008216909_decay457_trial002
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 43 min
current_train_loss: 0.0028572313
current_train_psnr: 31.208572
testset_mean_loss: 0.0013334632
testset_mean_psnr: 29.155115
testset_mean_ssim: 0.870486
testset_mean_lpips: 0.093213
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0010478157, psnr=29.797151, ssim=0.820846, lpips=0.133224
  image_001: loss=0.0012410575, psnr=29.062081, ssim=0.822196, lpips=0.138350
  image_002: loss=0.0010533682, psnr=29.774198, ssim=0.842039, lpips=0.126533
  image_003: loss=0.0013688387, psnr=28.636477, ssim=0.805248, lpips=0.148122
  image_004: loss=0.0011053123, psnr=29.565150, ssim=0.831900, lpips=0.108731
  image_005: loss=0.0009456158, psnr=30.242853, ssim=0.868470, lpips=0.093898
  image_006: loss=0.0013982990, psnr=28.543999, ssim=0.865096, lpips=0.097336
  image_007: loss=0.0011883072, psnr=29.250712, ssim=0.879159, lpips=0.082781
  image_008: loss=0.0010063630, psnr=29.972453, ssim=0.891700, lpips=0.073120
  image_009: loss=0.0009749007, psnr=30.110396, ssim=0.900964, lpips=0.061405
  image_010: loss=0.0008976128, psnr=30.469109, ssim=0.915324, lpips=0.049787
  image_011: loss=0.0020086968, psnr=26.970856, ssim=0.884431, lpips=0.070655
  image_012: loss=0.0044648149, psnr=23.501965, ssim=0.859272, lpips=0.102022
  image_013: loss=0.0026462779, psnr=25.773645, ssim=0.883393, lpips=0.081380
  image_014: loss=0.0015642826, psnr=28.056848, ssim=0.920754, lpips=0.050910
  image_015: loss=0.0010396721, psnr=29.831036, ssim=0.938886, lpips=0.045763
  image_016: loss=0.0009757282, psnr=30.106711, ssim=0.920464, lpips=0.051804
  image_017: loss=0.0010840519, psnr=29.649499, ssim=0.902653, lpips=0.062366
  image_018: loss=0.0015861725, psnr=27.996496, ssim=0.870472, lpips=0.095005
  image_019: loss=0.0012863701, psnr=28.906340, ssim=0.869390, lpips=0.093787
  image_020: loss=0.0010036952, psnr=29.983981, ssim=0.878750, lpips=0.096891
  image_021: loss=0.0009290966, psnr=30.319391, ssim=0.862313, lpips=0.101375
  image_022: loss=0.0007722701, psnr=31.122307, ssim=0.860311, lpips=0.107513
  image_023: loss=0.0007680708, psnr=31.145987, ssim=0.846987, lpips=0.124195
  image_024: loss=0.0009798890, psnr=30.088231, ssim=0.821141, lpips=0.133367
