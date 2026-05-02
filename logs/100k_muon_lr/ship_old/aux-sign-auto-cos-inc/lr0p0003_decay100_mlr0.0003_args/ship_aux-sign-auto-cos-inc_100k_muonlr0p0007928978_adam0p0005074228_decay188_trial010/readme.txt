====================================================================================================
saved_time: 2026-05-01 22:57:24
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ship.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p0007928978_adam0p0005074228_decay188_trial010 --optimizer aux-sign-auto-cos-inc --lrate 0.0005074227748601469 --lrate_decay 188 --muon_lrate 0.0007928978254663852 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p0007928978_adam0p0005074228_decay188_trial010/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p0007928978_adam0p0005074228_decay188_trial010
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 30 min
current_train_loss: 0.0027105021
current_train_psnr: 31.216278
testset_mean_loss: 0.0013109877
testset_mean_psnr: 29.037270
testset_mean_ssim: 0.863748
testset_mean_lpips: 0.105935
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0011305597, psnr=29.467065, ssim=0.807478, lpips=0.157921
  image_001: loss=0.0013222151, psnr=28.786979, ssim=0.812421, lpips=0.158543
  image_002: loss=0.0011025751, psnr=29.575918, ssim=0.835145, lpips=0.139569
  image_003: loss=0.0014297307, psnr=28.447457, ssim=0.794603, lpips=0.161329
  image_004: loss=0.0012043869, psnr=29.192339, ssim=0.821551, lpips=0.123507
  image_005: loss=0.0010117188, psnr=29.949402, ssim=0.862200, lpips=0.104778
  image_006: loss=0.0014568691, psnr=28.365794, ssim=0.860116, lpips=0.108370
  image_007: loss=0.0012533400, psnr=29.019310, ssim=0.871982, lpips=0.093688
  image_008: loss=0.0010669396, psnr=29.718601, ssim=0.884696, lpips=0.087809
  image_009: loss=0.0010792242, psnr=29.668883, ssim=0.892180, lpips=0.073475
  image_010: loss=0.0009483952, psnr=30.230106, ssim=0.908275, lpips=0.061106
  image_011: loss=0.0018901052, psnr=27.235140, ssim=0.880664, lpips=0.089483
  image_012: loss=0.0027933535, psnr=25.538741, ssim=0.868495, lpips=0.108744
  image_013: loss=0.0023477732, psnr=26.293438, ssim=0.877309, lpips=0.092872
  image_014: loss=0.0017116705, psnr=27.665798, ssim=0.912811, lpips=0.066240
  image_015: loss=0.0010948813, psnr=29.606329, ssim=0.932324, lpips=0.056260
  image_016: loss=0.0010142346, psnr=29.938615, ssim=0.914795, lpips=0.056293
  image_017: loss=0.0011875118, psnr=29.253620, ssim=0.895712, lpips=0.071871
  image_018: loss=0.0016731612, psnr=27.764622, ssim=0.866573, lpips=0.102194
  image_019: loss=0.0013586699, psnr=28.668860, ssim=0.861719, lpips=0.104531
  image_020: loss=0.0010694772, psnr=29.708284, ssim=0.871013, lpips=0.105236
  image_021: loss=0.0009968237, psnr=30.013816, ssim=0.852825, lpips=0.113923
  image_022: loss=0.0007989866, psnr=30.974605, ssim=0.854585, lpips=0.119082
  image_023: loss=0.0007875068, psnr=31.037457, ssim=0.841408, lpips=0.139615
  image_024: loss=0.0010445821, psnr=29.810574, ssim=0.812814, lpips=0.151940
