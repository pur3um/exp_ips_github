====================================================================================================
saved_time: 2026-05-02 05:09:43
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0019739948_adam0p0003498388_decay376_trial012 --optimizer aux-sign-auto-cos-inc --lrate 0.00034983882527239615 --lrate_decay 376 --muon_lrate 0.0019739948242649733 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0019739948_adam0p0003498388_decay376_trial012/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0019739948_adam0p0003498388_decay376_trial012
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 27 min
current_train_loss: 0.0024341010
current_train_psnr: 32.169048
testset_mean_loss: 0.0012273370
testset_mean_psnr: 29.257925
testset_mean_ssim: 0.966819
testset_mean_lpips: 0.023365
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008513587, psnr=30.698874, ssim=0.971284, lpips=0.021580
  image_001: loss=0.0013469907, psnr=28.706354, ssim=0.965739, lpips=0.025720
  image_002: loss=0.0015154700, psnr=28.194526, ssim=0.962174, lpips=0.026051
  image_003: loss=0.0017090075, psnr=27.672560, ssim=0.957845, lpips=0.027953
  image_004: loss=0.0018584916, psnr=27.308394, ssim=0.958354, lpips=0.031480
  image_005: loss=0.0012835462, psnr=28.915885, ssim=0.965623, lpips=0.026449
  image_006: loss=0.0012909370, psnr=28.890949, ssim=0.965940, lpips=0.024091
  image_007: loss=0.0008519734, psnr=30.695739, ssim=0.974271, lpips=0.016638
  image_008: loss=0.0012422317, psnr=29.057974, ssim=0.963026, lpips=0.026476
  image_009: loss=0.0015186671, psnr=28.185374, ssim=0.957745, lpips=0.027462
  image_010: loss=0.0009144014, psnr=30.388631, ssim=0.974112, lpips=0.018292
  image_011: loss=0.0008230431, psnr=30.845774, ssim=0.978292, lpips=0.015160
  image_012: loss=0.0008353783, psnr=30.781168, ssim=0.977272, lpips=0.016357
  image_013: loss=0.0007415332, psnr=31.298694, ssim=0.978905, lpips=0.014064
  image_014: loss=0.0011734081, psnr=29.305509, ssim=0.969085, lpips=0.020912
  image_015: loss=0.0010973603, psnr=29.596507, ssim=0.965800, lpips=0.023893
  image_016: loss=0.0011305868, psnr=29.466960, ssim=0.973650, lpips=0.017024
  image_017: loss=0.0010880057, psnr=29.633688, ssim=0.971956, lpips=0.021205
  image_018: loss=0.0012130144, psnr=29.161340, ssim=0.967715, lpips=0.022332
  image_019: loss=0.0012459296, psnr=29.045065, ssim=0.967219, lpips=0.022578
  image_020: loss=0.0014454946, psnr=28.399835, ssim=0.961333, lpips=0.029316
  image_021: loss=0.0008101235, psnr=30.914487, ssim=0.970776, lpips=0.021625
  image_022: loss=0.0013935287, psnr=28.558841, ssim=0.959609, lpips=0.026514
  image_023: loss=0.0018779200, psnr=27.263229, ssim=0.954247, lpips=0.034200
  image_024: loss=0.0014250235, psnr=28.461779, ssim=0.958497, lpips=0.026761
