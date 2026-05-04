====================================================================================================
saved_time: 2026-05-04 09:38:36
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0008998362_adam0p0006404984_decay365_trial018 --optimizer aux-sign-auto-cos-inc --lrate 0.0006404983857171204 --lrate_decay 365 --muon_lrate 0.000899836160616584 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0008998362_adam0p0006404984_decay365_trial018/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0008998362_adam0p0006404984_decay365_trial018
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0052891383
current_train_psnr: 27.903135
testset_mean_loss: 0.0030584000
testset_mean_psnr: 25.414445
testset_mean_ssim: 0.926783
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0018194497, psnr=27.400599, ssim=0.928396, lpips=unavailable
  image_001: loss=0.0022634263, psnr=26.452336, ssim=0.924199, lpips=unavailable
  image_002: loss=0.0014594977, psnr=28.357965, ssim=0.946086, lpips=unavailable
  image_003: loss=0.0022642983, psnr=26.450663, ssim=0.936183, lpips=unavailable
  image_004: loss=0.0020625654, psnr=26.855922, ssim=0.933702, lpips=unavailable
  image_005: loss=0.0018713871, psnr=27.278363, ssim=0.938317, lpips=unavailable
  image_006: loss=0.0047949138, psnr=23.192192, ssim=0.899627, lpips=unavailable
  image_007: loss=0.0037814644, psnr=24.223400, ssim=0.915237, lpips=unavailable
  image_008: loss=0.0031402681, psnr=25.030333, ssim=0.928487, lpips=unavailable
  image_009: loss=0.0041017830, psnr=23.870273, ssim=0.925617, lpips=unavailable
  image_010: loss=0.0036345071, psnr=24.395545, ssim=0.932094, lpips=unavailable
  image_011: loss=0.0042370474, psnr=23.729367, ssim=0.915395, lpips=unavailable
  image_012: loss=0.0040101451, psnr=23.968399, ssim=0.925894, lpips=unavailable
  image_013: loss=0.0028757148, psnr=25.412542, ssim=0.932352, lpips=unavailable
  image_014: loss=0.0026002030, psnr=25.849927, ssim=0.936312, lpips=unavailable
  image_015: loss=0.0041945321, psnr=23.773165, ssim=0.930208, lpips=unavailable
  image_016: loss=0.0022277574, psnr=26.521321, ssim=0.947976, lpips=unavailable
  image_017: loss=0.0028120393, psnr=25.509786, ssim=0.929897, lpips=unavailable
  image_018: loss=0.0032267072, psnr=24.912404, ssim=0.925539, lpips=unavailable
  image_019: loss=0.0055035818, psnr=22.593546, ssim=0.889419, lpips=unavailable
  image_020: loss=0.0034743538, psnr=24.591259, ssim=0.913441, lpips=unavailable
  image_021: loss=0.0040190271, psnr=23.958791, ssim=0.908509, lpips=unavailable
  image_022: loss=0.0025002055, psnr=26.020243, ssim=0.930353, lpips=unavailable
  image_023: loss=0.0020381315, psnr=26.907678, ssim=0.935097, lpips=unavailable
  image_024: loss=0.0015469933, psnr=28.105115, ssim=0.941226, lpips=unavailable
