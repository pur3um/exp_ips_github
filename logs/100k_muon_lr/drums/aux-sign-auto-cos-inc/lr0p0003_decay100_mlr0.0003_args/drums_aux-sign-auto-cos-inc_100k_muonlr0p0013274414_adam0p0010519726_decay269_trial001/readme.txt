====================================================================================================
saved_time: 2026-05-02 19:58:33
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0013274414_adam0p0010519726_decay269_trial001 --optimizer aux-sign-auto-cos-inc --lrate 0.0010519726230559959 --lrate_decay 269 --muon_lrate 0.0013274414292574908 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0013274414_adam0p0010519726_decay269_trial001/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0013274414_adam0p0010519726_decay269_trial001
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 10 min
current_train_loss: 0.0050327843
current_train_psnr: 28.207020
testset_mean_loss: 0.0029694768
testset_mean_psnr: 25.576083
testset_mean_ssim: 0.930030
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017184294, psnr=27.648683, ssim=0.931124, lpips=unavailable
  image_001: loss=0.0020728761, psnr=26.834266, ssim=0.928603, lpips=unavailable
  image_002: loss=0.0013239075, psnr=28.781423, ssim=0.949063, lpips=unavailable
  image_003: loss=0.0022522290, psnr=26.473874, ssim=0.937049, lpips=unavailable
  image_004: loss=0.0019014074, psnr=27.209248, ssim=0.937447, lpips=unavailable
  image_005: loss=0.0017585595, psnr=27.548429, ssim=0.942255, lpips=unavailable
  image_006: loss=0.0049549872, psnr=23.049575, ssim=0.903163, lpips=unavailable
  image_007: loss=0.0038463734, psnr=24.149485, ssim=0.917853, lpips=unavailable
  image_008: loss=0.0029212998, psnr=25.344239, ssim=0.933266, lpips=unavailable
  image_009: loss=0.0038842906, psnr=24.106883, ssim=0.930414, lpips=unavailable
  image_010: loss=0.0032238618, psnr=24.916236, ssim=0.938339, lpips=unavailable
  image_011: loss=0.0039177891, psnr=24.069589, ssim=0.920594, lpips=unavailable
  image_012: loss=0.0040681609, psnr=23.906019, ssim=0.927627, lpips=unavailable
  image_013: loss=0.0026638696, psnr=25.744870, ssim=0.937897, lpips=unavailable
  image_014: loss=0.0024668833, psnr=26.078514, ssim=0.940566, lpips=unavailable
  image_015: loss=0.0039666039, psnr=24.015812, ssim=0.934150, lpips=unavailable
  image_016: loss=0.0019558680, psnr=27.086604, ssim=0.952655, lpips=unavailable
  image_017: loss=0.0029399244, psnr=25.316638, ssim=0.932788, lpips=unavailable
  image_018: loss=0.0030637747, psnr=25.137432, ssim=0.930011, lpips=unavailable
  image_019: loss=0.0056705712, psnr=22.463732, ssim=0.890410, lpips=unavailable
  image_020: loss=0.0033819184, psnr=24.708369, ssim=0.915653, lpips=unavailable
  image_021: loss=0.0041518025, psnr=23.817633, ssim=0.910941, lpips=unavailable
  image_022: loss=0.0025569976, psnr=25.922697, ssim=0.931592, lpips=unavailable
  image_023: loss=0.0020769648, psnr=26.825708, ssim=0.934414, lpips=unavailable
  image_024: loss=0.0014975710, psnr=28.246125, ssim=0.942876, lpips=unavailable
