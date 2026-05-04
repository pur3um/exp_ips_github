====================================================================================================
saved_time: 2026-05-03 02:29:32
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0003533111_adam0p0011095984_decay471_trial004 --optimizer aux-sign-auto-cos-inc --lrate 0.0011095983846308643 --lrate_decay 471 --muon_lrate 0.00035331112522565144 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0003533111_adam0p0011095984_decay471_trial004/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0003533111_adam0p0011095984_decay471_trial004
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 9 min
current_train_loss: 0.0059793703
current_train_psnr: 26.807013
testset_mean_loss: 0.0035309524
testset_mean_psnr: 24.751537
testset_mean_ssim: 0.916524
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0022036047, psnr=26.568663, ssim=0.919043, lpips=unavailable
  image_001: loss=0.0026257101, psnr=25.807532, ssim=0.916561, lpips=unavailable
  image_002: loss=0.0019296117, psnr=27.145301, ssim=0.933960, lpips=unavailable
  image_003: loss=0.0025314013, psnr=25.966390, ssim=0.930463, lpips=unavailable
  image_004: loss=0.0023915647, psnr=26.213179, ssim=0.924549, lpips=unavailable
  image_005: loss=0.0023463031, psnr=26.296159, ssim=0.926859, lpips=unavailable
  image_006: loss=0.0053863116, psnr=22.687085, ssim=0.892190, lpips=unavailable
  image_007: loss=0.0044242418, psnr=23.541611, ssim=0.900404, lpips=unavailable
  image_008: loss=0.0038920594, psnr=24.098205, ssim=0.913541, lpips=unavailable
  image_009: loss=0.0045932117, psnr=23.378835, ssim=0.915849, lpips=unavailable
  image_010: loss=0.0043030581, psnr=23.662228, ssim=0.918061, lpips=unavailable
  image_011: loss=0.0048967409, psnr=23.100929, ssim=0.903152, lpips=unavailable
  image_012: loss=0.0041686078, psnr=23.800090, ssim=0.918044, lpips=unavailable
  image_013: loss=0.0033971348, psnr=24.688872, ssim=0.921403, lpips=unavailable
  image_014: loss=0.0030666357, psnr=25.133378, ssim=0.925770, lpips=unavailable
  image_015: loss=0.0046942355, psnr=23.284351, ssim=0.921981, lpips=unavailable
  image_016: loss=0.0026871781, psnr=25.707035, ssim=0.937475, lpips=unavailable
  image_017: loss=0.0032825777, psnr=24.837850, ssim=0.916337, lpips=unavailable
  image_018: loss=0.0038940054, psnr=24.096034, ssim=0.910486, lpips=unavailable
  image_019: loss=0.0061417213, psnr=22.117099, ssim=0.881001, lpips=unavailable
  image_020: loss=0.0040683909, psnr=23.905773, ssim=0.900330, lpips=unavailable
  image_021: loss=0.0043085404, psnr=23.656698, ssim=0.902002, lpips=unavailable
  image_022: loss=0.0028648465, psnr=25.428986, ssim=0.921648, lpips=unavailable
  image_023: loss=0.0023346436, psnr=26.317794, ssim=0.927634, lpips=unavailable
  image_024: loss=0.0018414721, psnr=27.348348, ssim=0.934345, lpips=unavailable
