====================================================================================================
saved_time: 2026-05-03 22:29:20
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p000652304_adam0p0016620521_decay225_trial013 --optimizer aux-sign-auto-cos-inc --lrate 0.001662052070515544 --lrate_decay 225 --muon_lrate 0.0006523039702489399 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p000652304_adam0p0016620521_decay225_trial013/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p000652304_adam0p0016620521_decay225_trial013
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0049748151
current_train_psnr: 27.628529
testset_mean_loss: 0.0031404160
testset_mean_psnr: 25.287362
testset_mean_ssim: 0.924781
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0018563154, psnr=27.313482, ssim=0.926060, lpips=unavailable
  image_001: loss=0.0023861527, psnr=26.223017, ssim=0.922091, lpips=unavailable
  image_002: loss=0.0015342962, psnr=28.140908, ssim=0.943915, lpips=unavailable
  image_003: loss=0.0023438539, psnr=26.300694, ssim=0.935033, lpips=unavailable
  image_004: loss=0.0021040167, psnr=26.769508, ssim=0.931846, lpips=unavailable
  image_005: loss=0.0019695819, psnr=27.056259, ssim=0.936596, lpips=unavailable
  image_006: loss=0.0049094702, psnr=23.089654, ssim=0.897915, lpips=unavailable
  image_007: loss=0.0040508471, psnr=23.924541, ssim=0.911141, lpips=unavailable
  image_008: loss=0.0032476787, psnr=24.884269, ssim=0.925737, lpips=unavailable
  image_009: loss=0.0041560335, psnr=23.813209, ssim=0.923435, lpips=unavailable
  image_010: loss=0.0036405562, psnr=24.388322, ssim=0.931139, lpips=unavailable
  image_011: loss=0.0042372718, psnr=23.729137, ssim=0.914256, lpips=unavailable
  image_012: loss=0.0040937662, psnr=23.878770, ssim=0.922721, lpips=unavailable
  image_013: loss=0.0028593573, psnr=25.437316, ssim=0.932658, lpips=unavailable
  image_014: loss=0.0026347737, psnr=25.792567, ssim=0.934804, lpips=unavailable
  image_015: loss=0.0041204593, psnr=23.850544, ssim=0.931002, lpips=unavailable
  image_016: loss=0.0022984690, psnr=26.385613, ssim=0.946070, lpips=unavailable
  image_017: loss=0.0031072651, psnr=25.076217, ssim=0.926278, lpips=unavailable
  image_018: loss=0.0032424305, psnr=24.891293, ssim=0.923007, lpips=unavailable
  image_019: loss=0.0055967956, psnr=22.520605, ssim=0.885854, lpips=unavailable
  image_020: loss=0.0035839942, psnr=24.456327, ssim=0.909658, lpips=unavailable
  image_021: loss=0.0041766767, psnr=23.791691, ssim=0.907649, lpips=unavailable
  image_022: loss=0.0026480858, psnr=25.770679, ssim=0.927885, lpips=unavailable
  image_023: loss=0.0020934406, psnr=26.791393, ssim=0.933102, lpips=unavailable
  image_024: loss=0.0016188123, psnr=27.908035, ssim=0.939671, lpips=unavailable
