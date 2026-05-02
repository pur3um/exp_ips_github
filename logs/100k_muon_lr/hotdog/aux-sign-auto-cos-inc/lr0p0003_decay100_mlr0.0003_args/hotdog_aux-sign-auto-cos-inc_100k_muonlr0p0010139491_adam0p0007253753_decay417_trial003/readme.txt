====================================================================================================
saved_time: 2026-05-01 03:31:54
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0010139491_adam0p0007253753_decay417_trial003 --optimizer aux-sign-auto-cos-inc --lrate 0.0007253753172998836 --lrate_decay 417 --muon_lrate 0.0010139491476424019 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0010139491_adam0p0007253753_decay417_trial003/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0010139491_adam0p0007253753_decay417_trial003
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 30 min
current_train_loss: 0.0010358011
current_train_psnr: 35.476631
testset_mean_loss: 0.0002549970
testset_mean_psnr: 36.798868
testset_mean_ssim: 0.979550
testset_mean_lpips: 0.018924
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0002152980, psnr=36.669598, ssim=0.980326, lpips=0.016848
  image_001: loss=0.0001897075, psnr=37.219152, ssim=0.982438, lpips=0.020078
  image_002: loss=0.0002462560, psnr=36.086130, ssim=0.978136, lpips=0.020757
  image_003: loss=0.0001673202, psnr=37.764514, ssim=0.981074, lpips=0.020460
  image_004: loss=0.0001229984, psnr=39.101002, ssim=0.983993, lpips=0.015297
  image_005: loss=0.0001708008, psnr=37.675098, ssim=0.982678, lpips=0.013243
  image_006: loss=0.0002336901, psnr=36.313596, ssim=0.981073, lpips=0.014425
  image_007: loss=0.0001489042, psnr=38.270927, ssim=0.984488, lpips=0.013956
  image_008: loss=0.0001600309, psnr=37.957959, ssim=0.979169, lpips=0.014368
  image_009: loss=0.0001862506, psnr=37.299021, ssim=0.972190, lpips=0.024903
  image_010: loss=0.0001611011, psnr=37.929011, ssim=0.974268, lpips=0.025480
  image_011: loss=0.0002634020, psnr=35.793807, ssim=0.973023, lpips=0.024938
  image_012: loss=0.0002613965, psnr=35.827000, ssim=0.982664, lpips=0.012931
  image_013: loss=0.0001692533, psnr=37.714626, ssim=0.988211, lpips=0.010632
  image_014: loss=0.0002966296, psnr=35.277853, ssim=0.982385, lpips=0.016588
  image_015: loss=0.0014545569, psnr=28.372692, ssim=0.962629, lpips=0.032680
  image_016: loss=0.0005529560, psnr=32.573093, ssim=0.968780, lpips=0.029821
  image_017: loss=0.0001629053, psnr=37.880646, ssim=0.982015, lpips=0.016608
  image_018: loss=0.0002459833, psnr=36.090942, ssim=0.979046, lpips=0.014251
  image_019: loss=0.0001977309, psnr=37.039253, ssim=0.982432, lpips=0.013404
  image_020: loss=0.0001252456, psnr=39.022370, ssim=0.985585, lpips=0.013299
  image_021: loss=0.0001298198, psnr=38.866587, ssim=0.981814, lpips=0.015163
  image_022: loss=0.0001419965, psnr=38.477220, ssim=0.979829, lpips=0.024588
  image_023: loss=0.0001561352, psnr=38.064989, ssim=0.980526, lpips=0.022709
  image_024: loss=0.0002145551, psnr=36.684609, ssim=0.979970, lpips=0.025679
