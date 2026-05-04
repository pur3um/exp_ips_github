====================================================================================================
saved_time: 2026-05-04 00:43:07
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0018529625_adam0p0015288108_decay315_trial014 --optimizer aux-sign-auto-cos-inc --lrate 0.0015288107660314468 --lrate_decay 315 --muon_lrate 0.0018529625116746901 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0018529625_adam0p0015288108_decay315_trial014/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0018529625_adam0p0015288108_decay315_trial014
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0045027407
current_train_psnr: 29.006050
testset_mean_loss: 0.0028915875
testset_mean_psnr: 25.694263
testset_mean_ssim: 0.931188
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0016769720, psnr=27.754742, ssim=0.932867, lpips=unavailable
  image_001: loss=0.0020695650, psnr=26.841209, ssim=0.930228, lpips=unavailable
  image_002: loss=0.0012392208, psnr=29.068513, ssim=0.950667, lpips=unavailable
  image_003: loss=0.0021011035, psnr=26.775525, ssim=0.940643, lpips=unavailable
  image_004: loss=0.0019556601, psnr=27.087066, ssim=0.935878, lpips=unavailable
  image_005: loss=0.0017724668, psnr=27.514219, ssim=0.941621, lpips=unavailable
  image_006: loss=0.0046440875, psnr=23.330996, ssim=0.905875, lpips=unavailable
  image_007: loss=0.0035384025, psnr=24.511928, ssim=0.919211, lpips=unavailable
  image_008: loss=0.0028111455, psnr=25.511167, ssim=0.936048, lpips=unavailable
  image_009: loss=0.0035794573, psnr=24.461828, ssim=0.931142, lpips=unavailable
  image_010: loss=0.0031691296, psnr=24.990600, ssim=0.938904, lpips=unavailable
  image_011: loss=0.0038559742, psnr=24.138659, ssim=0.920604, lpips=unavailable
  image_012: loss=0.0044810222, psnr=23.486229, ssim=0.926488, lpips=unavailable
  image_013: loss=0.0024971711, psnr=26.025517, ssim=0.939611, lpips=unavailable
  image_014: loss=0.0023910340, psnr=26.214142, ssim=0.942825, lpips=unavailable
  image_015: loss=0.0041633225, psnr=23.805599, ssim=0.931476, lpips=unavailable
  image_016: loss=0.0019181011, psnr=27.171285, ssim=0.952954, lpips=unavailable
  image_017: loss=0.0027434845, psnr=25.616975, ssim=0.933720, lpips=unavailable
  image_018: loss=0.0029602391, psnr=25.286732, ssim=0.932215, lpips=unavailable
  image_019: loss=0.0055333306, psnr=22.570134, ssim=0.891741, lpips=unavailable
  image_020: loss=0.0033533138, psnr=24.745258, ssim=0.916948, lpips=unavailable
  image_021: loss=0.0039123143, psnr=24.075663, ssim=0.913301, lpips=unavailable
  image_022: loss=0.0023887511, psnr=26.218291, ssim=0.933490, lpips=unavailable
  image_023: loss=0.0020338986, psnr=26.916707, ssim=0.937176, lpips=unavailable
  image_024: loss=0.0015005186, psnr=28.237586, ssim=0.944078, lpips=unavailable
