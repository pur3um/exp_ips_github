====================================================================================================
saved_time: 2026-05-02 22:08:42
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0027592053_adam0p0008216909_decay457_trial002 --optimizer aux-sign-auto-cos-inc --lrate 0.0008216908785704095 --lrate_decay 457 --muon_lrate 0.0027592052977704327 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0027592053_adam0p0008216909_decay457_trial002/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0027592053_adam0p0008216909_decay457_trial002
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 9 min
current_train_loss: 0.0049834992
current_train_psnr: 28.254747
testset_mean_loss: 0.0028265930
testset_mean_psnr: 25.741576
testset_mean_ssim: 0.932035
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017196066, psnr=27.645709, ssim=0.933759, lpips=unavailable
  image_001: loss=0.0021770459, psnr=26.621324, ssim=0.928870, lpips=unavailable
  image_002: loss=0.0012778005, psnr=28.935369, ssim=0.950935, lpips=unavailable
  image_003: loss=0.0023029216, psnr=26.377208, ssim=0.938792, lpips=unavailable
  image_004: loss=0.0018273187, psnr=27.381857, ssim=0.938591, lpips=unavailable
  image_005: loss=0.0017599618, psnr=27.544967, ssim=0.943510, lpips=unavailable
  image_006: loss=0.0040007639, psnr=23.978571, ssim=0.909188, lpips=unavailable
  image_007: loss=0.0034916641, psnr=24.569675, ssim=0.921914, lpips=unavailable
  image_008: loss=0.0028280560, psnr=25.485120, ssim=0.934610, lpips=unavailable
  image_009: loss=0.0035914823, psnr=24.447263, ssim=0.932754, lpips=unavailable
  image_010: loss=0.0033801766, psnr=24.710606, ssim=0.937670, lpips=unavailable
  image_011: loss=0.0040922398, psnr=23.880389, ssim=0.920500, lpips=unavailable
  image_012: loss=0.0037826446, psnr=24.222044, ssim=0.928059, lpips=unavailable
  image_013: loss=0.0025288279, psnr=25.970807, ssim=0.939204, lpips=unavailable
  image_014: loss=0.0024118745, psnr=26.176453, ssim=0.941661, lpips=unavailable
  image_015: loss=0.0039251605, psnr=24.061426, ssim=0.934204, lpips=unavailable
  image_016: loss=0.0019710786, psnr=27.052960, ssim=0.953341, lpips=unavailable
  image_017: loss=0.0026624328, psnr=25.747213, ssim=0.936238, lpips=unavailable
  image_018: loss=0.0029850756, psnr=25.250447, ssim=0.931514, lpips=unavailable
  image_019: loss=0.0047994405, psnr=23.188094, ssim=0.895217, lpips=unavailable
  image_020: loss=0.0032958188, psnr=24.820367, ssim=0.920029, lpips=unavailable
  image_021: loss=0.0038854710, psnr=24.105563, ssim=0.914425, lpips=unavailable
  image_022: loss=0.0025369567, psnr=25.956869, ssim=0.933477, lpips=unavailable
  image_023: loss=0.0019703791, psnr=27.054502, ssim=0.938074, lpips=unavailable
  image_024: loss=0.0014606260, psnr=28.354609, ssim=0.944345, lpips=unavailable
