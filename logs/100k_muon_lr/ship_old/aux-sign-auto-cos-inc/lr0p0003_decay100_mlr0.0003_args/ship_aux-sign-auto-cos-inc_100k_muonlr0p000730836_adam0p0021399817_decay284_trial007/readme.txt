====================================================================================================
saved_time: 2026-05-01 12:26:08
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p000730836_adam0p0021399817_decay284_trial007 --optimizer aux-sign-auto-cos-inc --lrate 0.0021399816882859213 --lrate_decay 284 --muon_lrate 0.0007308359664212255 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p000730836_adam0p0021399817_decay284_trial007/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p000730836_adam0p0021399817_decay284_trial007
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 32 min
current_train_loss: 0.0029137132
current_train_psnr: 30.609997
testset_mean_loss: 0.0013567459
testset_mean_psnr: 28.952292
testset_mean_ssim: 0.862133
testset_mean_lpips: 0.105762
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0011338128, psnr=29.454586, ssim=0.808982, lpips=0.153588
  image_001: loss=0.0013115066, psnr=28.822295, ssim=0.814487, lpips=0.155431
  image_002: loss=0.0011040185, psnr=29.570236, ssim=0.835865, lpips=0.141035
  image_003: loss=0.0014463990, psnr=28.397119, ssim=0.793847, lpips=0.158998
  image_004: loss=0.0011902419, psnr=29.243647, ssim=0.821037, lpips=0.117701
  image_005: loss=0.0010146032, psnr=29.937037, ssim=0.861369, lpips=0.101353
  image_006: loss=0.0015147516, psnr=28.196586, ssim=0.852514, lpips=0.113698
  image_007: loss=0.0012526968, psnr=29.021540, ssim=0.869648, lpips=0.095382
  image_008: loss=0.0010540708, psnr=29.771302, ssim=0.884683, lpips=0.086223
  image_009: loss=0.0010570468, psnr=29.759057, ssim=0.892327, lpips=0.074545
  image_010: loss=0.0009697627, psnr=30.133345, ssim=0.906033, lpips=0.061029
  image_011: loss=0.0020011747, psnr=26.987150, ssim=0.877321, lpips=0.084458
  image_012: loss=0.0033398401, psnr=24.762743, ssim=0.856586, lpips=0.108581
  image_013: loss=0.0026137438, psnr=25.827370, ssim=0.874961, lpips=0.093433
  image_014: loss=0.0017297727, psnr=27.620109, ssim=0.912570, lpips=0.069270
  image_015: loss=0.0010916059, psnr=29.619341, ssim=0.932406, lpips=0.054806
  image_016: loss=0.0010085880, psnr=29.962862, ssim=0.914687, lpips=0.056673
  image_017: loss=0.0011867102, psnr=29.256553, ssim=0.894785, lpips=0.070008
  image_018: loss=0.0018076409, psnr=27.428878, ssim=0.859297, lpips=0.110059
  image_019: loss=0.0013532388, psnr=28.686255, ssim=0.860049, lpips=0.106450
  image_020: loss=0.0010740663, psnr=29.689689, ssim=0.870928, lpips=0.103286
  image_021: loss=0.0009970961, psnr=30.012630, ssim=0.853136, lpips=0.112323
  image_022: loss=0.0008184647, psnr=30.870000, ssim=0.853504, lpips=0.115908
  image_023: loss=0.0007919892, psnr=31.012807, ssim=0.841031, lpips=0.140009
  image_024: loss=0.0010558063, psnr=29.764157, ssim=0.811276, lpips=0.159793
