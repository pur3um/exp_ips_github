====================================================================================================
saved_time: 2026-04-30 21:58:57
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial003 --optimizer aux-sign-auto-cos-inc --lrate 0.0007253753172998836 --lrate_decay 417 --muon_lrate 0.001328467245876384 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial003/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial003
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0024390249
current_train_psnr: 32.556755
testset_mean_loss: 0.0012958111
testset_mean_psnr: 29.072657
testset_mean_ssim: 0.965829
testset_mean_lpips: 0.024766
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008891741, psnr=30.510131, ssim=0.970164, lpips=0.023017
  image_001: loss=0.0013675505, psnr=28.640566, ssim=0.964216, lpips=0.026921
  image_002: loss=0.0016036185, psnr=27.948989, ssim=0.960229, lpips=0.028544
  image_003: loss=0.0018194759, psnr=27.400537, ssim=0.954365, lpips=0.032578
  image_004: loss=0.0018339092, psnr=27.366221, ssim=0.957021, lpips=0.032115
  image_005: loss=0.0012798711, psnr=28.928337, ssim=0.966058, lpips=0.025644
  image_006: loss=0.0013438006, psnr=28.716651, ssim=0.967013, lpips=0.026127
  image_007: loss=0.0008522163, psnr=30.694501, ssim=0.974449, lpips=0.017470
  image_008: loss=0.0013920349, psnr=28.563498, ssim=0.960628, lpips=0.027834
  image_009: loss=0.0015075278, psnr=28.217346, ssim=0.957650, lpips=0.030357
  image_010: loss=0.0009576913, psnr=30.187744, ssim=0.973047, lpips=0.018391
  image_011: loss=0.0008283688, psnr=30.817762, ssim=0.977872, lpips=0.014361
  image_012: loss=0.0008472985, psnr=30.719635, ssim=0.976849, lpips=0.016756
  image_013: loss=0.0007561927, psnr=31.213675, ssim=0.978508, lpips=0.016673
  image_014: loss=0.0012398176, psnr=29.066422, ssim=0.968369, lpips=0.022362
  image_015: loss=0.0010985848, psnr=29.591664, ssim=0.965283, lpips=0.022779
  image_016: loss=0.0011546898, psnr=29.375346, ssim=0.972752, lpips=0.018217
  image_017: loss=0.0010791417, psnr=29.669215, ssim=0.972519, lpips=0.022187
  image_018: loss=0.0012739367, psnr=28.948521, ssim=0.968289, lpips=0.022414
  image_019: loss=0.0012737209, psnr=28.949257, ssim=0.966771, lpips=0.022659
  image_020: loss=0.0015172592, psnr=28.189402, ssim=0.959860, lpips=0.027973
  image_021: loss=0.0008451804, psnr=30.730506, ssim=0.968826, lpips=0.024543
  image_022: loss=0.0014115506, psnr=28.503035, ssim=0.957938, lpips=0.030901
  image_023: loss=0.0027057228, psnr=25.677167, ssim=0.949188, lpips=0.040034
  image_024: loss=0.0015169438, psnr=28.190305, ssim=0.957851, lpips=0.028301
