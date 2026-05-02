====================================================================================================
saved_time: 2026-05-01 05:35:05
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0003636926_adam0p0003141717_decay227_trial008 --optimizer aux-sign-auto-cos-inc --lrate 0.00031417172594439365 --lrate_decay 227 --muon_lrate 0.0003636925892547774 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0003636926_adam0p0003141717_decay227_trial008/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0003636926_adam0p0003141717_decay227_trial008
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0020110761
current_train_psnr: 32.816662
testset_mean_loss: 0.0006402339
testset_mean_psnr: 32.030684
testset_mean_ssim: 0.972226
testset_mean_lpips: 0.037916
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0005881734, psnr=32.304945, ssim=0.971831, lpips=0.027977
  image_001: loss=0.0004340485, psnr=33.624616, ssim=0.975319, lpips=0.036620
  image_002: loss=0.0007180575, psnr=31.438407, ssim=0.969615, lpips=0.041609
  image_003: loss=0.0006053250, psnr=32.180113, ssim=0.969716, lpips=0.042646
  image_004: loss=0.0005701365, psnr=32.440211, ssim=0.968300, lpips=0.045838
  image_005: loss=0.0005468921, psnr=32.620983, ssim=0.971230, lpips=0.042380
  image_006: loss=0.0006377620, psnr=31.953413, ssim=0.977378, lpips=0.029354
  image_007: loss=0.0006772890, psnr=31.692259, ssim=0.971628, lpips=0.036284
  image_008: loss=0.0006780051, psnr=31.687670, ssim=0.965154, lpips=0.050964
  image_009: loss=0.0005838793, psnr=32.336768, ssim=0.970162, lpips=0.038121
  image_010: loss=0.0004136648, psnr=33.833514, ssim=0.976869, lpips=0.028461
  image_011: loss=0.0004623402, psnr=33.350383, ssim=0.979429, lpips=0.028174
  image_012: loss=0.0011321505, psnr=29.460958, ssim=0.976562, lpips=0.022975
  image_013: loss=0.0007834451, psnr=31.059914, ssim=0.979955, lpips=0.017381
  image_014: loss=0.0006467013, psnr=31.892962, ssim=0.977046, lpips=0.026469
  image_015: loss=0.0005945964, psnr=32.257777, ssim=0.975727, lpips=0.029477
  image_016: loss=0.0006221271, psnr=32.061208, ssim=0.972449, lpips=0.031938
  image_017: loss=0.0006329410, psnr=31.986367, ssim=0.970513, lpips=0.037184
  image_018: loss=0.0005127763, psnr=32.900720, ssim=0.975112, lpips=0.031704
  image_019: loss=0.0008104700, psnr=30.912630, ssim=0.976279, lpips=0.027700
  image_020: loss=0.0007251750, psnr=31.395571, ssim=0.965169, lpips=0.060296
  image_021: loss=0.0006687139, psnr=31.747596, ssim=0.965153, lpips=0.059716
  image_022: loss=0.0006722733, psnr=31.724540, ssim=0.967012, lpips=0.056415
  image_023: loss=0.0007354078, psnr=31.334717, ssim=0.966407, lpips=0.053634
  image_024: loss=0.0005534956, psnr=32.568857, ssim=0.971644, lpips=0.044590
