====================================================================================================
saved_time: 2026-05-03 18:02:11
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0016351994_adam0p0017548572_decay199_trial011 --optimizer aux-sign-auto-cos-inc --lrate 0.0017548572377658303 --lrate_decay 199 --muon_lrate 0.001635199446373098 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0016351994_adam0p0017548572_decay199_trial011/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0016351994_adam0p0017548572_decay199_trial011
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0044366219
current_train_psnr: 28.987921
testset_mean_loss: 0.0029271958
testset_mean_psnr: 25.669925
testset_mean_ssim: 0.931265
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0016536626, psnr=27.815531, ssim=0.933160, lpips=unavailable
  image_001: loss=0.0020715427, psnr=26.837061, ssim=0.929384, lpips=unavailable
  image_002: loss=0.0012490851, psnr=29.034079, ssim=0.951014, lpips=unavailable
  image_003: loss=0.0022606095, psnr=26.457744, ssim=0.939615, lpips=unavailable
  image_004: loss=0.0018820763, psnr=27.253628, ssim=0.936863, lpips=unavailable
  image_005: loss=0.0017404914, psnr=27.593281, ssim=0.942454, lpips=unavailable
  image_006: loss=0.0054604267, psnr=22.627734, ssim=0.903332, lpips=unavailable
  image_007: loss=0.0036251824, psnr=24.406701, ssim=0.920140, lpips=unavailable
  image_008: loss=0.0027815124, psnr=25.557190, ssim=0.936097, lpips=unavailable
  image_009: loss=0.0035051748, psnr=24.552903, ssim=0.932680, lpips=unavailable
  image_010: loss=0.0034038557, psnr=24.680288, ssim=0.937177, lpips=unavailable
  image_011: loss=0.0039704442, psnr=24.011609, ssim=0.920790, lpips=unavailable
  image_012: loss=0.0036829242, psnr=24.338072, ssim=0.930498, lpips=unavailable
  image_013: loss=0.0025183735, psnr=25.988798, ssim=0.939847, lpips=unavailable
  image_014: loss=0.0024326644, psnr=26.139178, ssim=0.941446, lpips=unavailable
  image_015: loss=0.0041163890, psnr=23.854836, ssim=0.931727, lpips=unavailable
  image_016: loss=0.0019638538, psnr=27.068908, ssim=0.951935, lpips=unavailable
  image_017: loss=0.0027308867, psnr=25.636963, ssim=0.934065, lpips=unavailable
  image_018: loss=0.0029587646, psnr=25.288896, ssim=0.930731, lpips=unavailable
  image_019: loss=0.0060433461, psnr=22.187225, ssim=0.890987, lpips=unavailable
  image_020: loss=0.0033534386, psnr=24.745096, ssim=0.918017, lpips=unavailable
  image_021: loss=0.0039440240, psnr=24.040604, ssim=0.914486, lpips=unavailable
  image_022: loss=0.0024307675, psnr=26.142566, ssim=0.932555, lpips=unavailable
  image_023: loss=0.0019557313, psnr=27.086908, ssim=0.937887, lpips=unavailable
  image_024: loss=0.0014446673, psnr=28.402321, ssim=0.944732, lpips=unavailable
