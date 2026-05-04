====================================================================================================
saved_time: 2026-05-03 11:21:28
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0017671529_adam0p002174216_decay121_trial008 --optimizer aux-sign-auto-cos-inc --lrate 0.0021742159536866593 --lrate_decay 121 --muon_lrate 0.00176715285719548 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0017671529_adam0p002174216_decay121_trial008/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0017671529_adam0p002174216_decay121_trial008
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0047025634
current_train_psnr: 28.578167
testset_mean_loss: 0.0028791365
testset_mean_psnr: 25.698208
testset_mean_ssim: 0.931001
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0016784377, psnr=27.750948, ssim=0.932267, lpips=unavailable
  image_001: loss=0.0021576134, psnr=26.660263, ssim=0.929110, lpips=unavailable
  image_002: loss=0.0012683287, psnr=28.967681, ssim=0.950513, lpips=unavailable
  image_003: loss=0.0021727646, psnr=26.629873, ssim=0.938548, lpips=unavailable
  image_004: loss=0.0019006440, psnr=27.210992, ssim=0.935817, lpips=unavailable
  image_005: loss=0.0017469527, psnr=27.577188, ssim=0.942575, lpips=unavailable
  image_006: loss=0.0049095610, psnr=23.089573, ssim=0.901353, lpips=unavailable
  image_007: loss=0.0036503412, psnr=24.376665, ssim=0.919137, lpips=unavailable
  image_008: loss=0.0028069527, psnr=25.517649, ssim=0.936076, lpips=unavailable
  image_009: loss=0.0036617317, psnr=24.363135, ssim=0.931562, lpips=unavailable
  image_010: loss=0.0032651664, psnr=24.860947, ssim=0.938728, lpips=unavailable
  image_011: loss=0.0039392207, psnr=24.045897, ssim=0.919649, lpips=unavailable
  image_012: loss=0.0037111207, psnr=24.304949, ssim=0.930414, lpips=unavailable
  image_013: loss=0.0025675450, psnr=25.904819, ssim=0.938940, lpips=unavailable
  image_014: loss=0.0024410826, psnr=26.124175, ssim=0.940742, lpips=unavailable
  image_015: loss=0.0039076372, psnr=24.080858, ssim=0.933876, lpips=unavailable
  image_016: loss=0.0019166988, psnr=27.174461, ssim=0.953256, lpips=unavailable
  image_017: loss=0.0027225157, psnr=25.650296, ssim=0.934317, lpips=unavailable
  image_018: loss=0.0029774944, psnr=25.261490, ssim=0.932018, lpips=unavailable
  image_019: loss=0.0053037172, psnr=22.754196, ssim=0.890948, lpips=unavailable
  image_020: loss=0.0033740010, psnr=24.718548, ssim=0.916837, lpips=unavailable
  image_021: loss=0.0039744303, psnr=24.007251, ssim=0.913180, lpips=unavailable
  image_022: loss=0.0024151641, psnr=26.170533, ssim=0.933570, lpips=unavailable
  image_023: loss=0.0020634059, psnr=26.854153, ssim=0.936773, lpips=unavailable
  image_024: loss=0.0014458839, psnr=28.398666, ssim=0.944818, lpips=unavailable
