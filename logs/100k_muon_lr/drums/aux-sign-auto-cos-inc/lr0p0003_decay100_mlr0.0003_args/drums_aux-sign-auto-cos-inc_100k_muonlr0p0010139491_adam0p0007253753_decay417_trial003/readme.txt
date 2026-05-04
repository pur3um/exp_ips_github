====================================================================================================
saved_time: 2026-05-03 00:19:24
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0010139491_adam0p0007253753_decay417_trial003 --optimizer aux-sign-auto-cos-inc --lrate 0.0007253753172998836 --lrate_decay 417 --muon_lrate 0.0010139491476424019 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0010139491_adam0p0007253753_decay417_trial003/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0010139491_adam0p0007253753_decay417_trial003
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 10 min
current_train_loss: 0.0048837769
current_train_psnr: 27.899611
testset_mean_loss: 0.0030317135
testset_mean_psnr: 25.470987
testset_mean_ssim: 0.928143
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017901141, psnr=27.471193, ssim=0.929841, lpips=unavailable
  image_001: loss=0.0022574503, psnr=26.463818, ssim=0.925200, lpips=unavailable
  image_002: loss=0.0013949550, psnr=28.554398, ssim=0.946932, lpips=unavailable
  image_003: loss=0.0021199696, psnr=26.736704, ssim=0.938864, lpips=unavailable
  image_004: loss=0.0020466775, psnr=26.889506, ssim=0.934027, lpips=unavailable
  image_005: loss=0.0018331720, psnr=27.367967, ssim=0.940583, lpips=unavailable
  image_006: loss=0.0052678380, psnr=22.783676, ssim=0.902889, lpips=unavailable
  image_007: loss=0.0036094466, psnr=24.425594, ssim=0.917290, lpips=unavailable
  image_008: loss=0.0030837499, psnr=25.109208, ssim=0.929327, lpips=unavailable
  image_009: loss=0.0040848195, psnr=23.888271, ssim=0.925520, lpips=unavailable
  image_010: loss=0.0033941772, psnr=24.692655, ssim=0.934352, lpips=unavailable
  image_011: loss=0.0043621771, psnr=23.602967, ssim=0.914466, lpips=unavailable
  image_012: loss=0.0037790297, psnr=24.226197, ssim=0.927725, lpips=unavailable
  image_013: loss=0.0027746190, psnr=25.567966, ssim=0.934562, lpips=unavailable
  image_014: loss=0.0025697700, psnr=25.901057, ssim=0.938124, lpips=unavailable
  image_015: loss=0.0040639215, psnr=23.910547, ssim=0.931477, lpips=unavailable
  image_016: loss=0.0021369494, psnr=26.702057, ssim=0.949515, lpips=unavailable
  image_017: loss=0.0029184965, psnr=25.348408, ssim=0.930758, lpips=unavailable
  image_018: loss=0.0031587158, psnr=25.004894, ssim=0.927606, lpips=unavailable
  image_019: loss=0.0055243964, psnr=22.577152, ssim=0.892957, lpips=unavailable
  image_020: loss=0.0034597206, psnr=24.609590, ssim=0.913753, lpips=unavailable
  image_021: loss=0.0040817531, psnr=23.891533, ssim=0.909765, lpips=unavailable
  image_022: loss=0.0025017960, psnr=26.017481, ssim=0.932475, lpips=unavailable
  image_023: loss=0.0020412561, psnr=26.901025, ssim=0.934200, lpips=unavailable
  image_024: loss=0.0015378670, psnr=28.130812, ssim=0.941371, lpips=unavailable
