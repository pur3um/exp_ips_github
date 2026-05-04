====================================================================================================
saved_time: 2026-05-03 13:34:25
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0006160331_adam0p0029945814_decay113_trial009 --optimizer aux-sign-auto-cos-inc --lrate 0.0029945813579718306 --lrate_decay 113 --muon_lrate 0.0006160330959166842 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0006160331_adam0p0029945814_decay113_trial009/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0006160331_adam0p0029945814_decay113_trial009
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 12 min
current_train_loss: 0.0049111350
current_train_psnr: 28.112591
testset_mean_loss: 0.0031459184
testset_mean_psnr: 25.297950
testset_mean_ssim: 0.924757
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0018792789, psnr=27.260087, ssim=0.925809, lpips=unavailable
  image_001: loss=0.0023665847, psnr=26.258779, ssim=0.921407, lpips=unavailable
  image_002: loss=0.0015049644, psnr=28.224737, ssim=0.944191, lpips=unavailable
  image_003: loss=0.0023331665, psnr=26.320542, ssim=0.935473, lpips=unavailable
  image_004: loss=0.0021077951, psnr=26.761716, ssim=0.931175, lpips=unavailable
  image_005: loss=0.0019420161, psnr=27.117472, ssim=0.937383, lpips=unavailable
  image_006: loss=0.0048810318, psnr=23.114884, ssim=0.899385, lpips=unavailable
  image_007: loss=0.0040486539, psnr=23.926893, ssim=0.908721, lpips=unavailable
  image_008: loss=0.0030979842, psnr=25.089208, ssim=0.927660, lpips=unavailable
  image_009: loss=0.0041277665, psnr=23.842849, ssim=0.924008, lpips=unavailable
  image_010: loss=0.0035673385, psnr=24.476557, ssim=0.931500, lpips=unavailable
  image_011: loss=0.0041922969, psnr=23.775480, ssim=0.913526, lpips=unavailable
  image_012: loss=0.0041968408, psnr=23.770775, ssim=0.922888, lpips=unavailable
  image_013: loss=0.0028618481, psnr=25.433534, ssim=0.932972, lpips=unavailable
  image_014: loss=0.0026141151, psnr=25.826753, ssim=0.935751, lpips=unavailable
  image_015: loss=0.0041924771, psnr=23.775293, ssim=0.929416, lpips=unavailable
  image_016: loss=0.0021882078, psnr=26.599114, ssim=0.947262, lpips=unavailable
  image_017: loss=0.0029687418, psnr=25.274276, ssim=0.927024, lpips=unavailable
  image_018: loss=0.0032411704, psnr=24.892981, ssim=0.923052, lpips=unavailable
  image_019: loss=0.0059751514, psnr=22.236511, ssim=0.883880, lpips=unavailable
  image_020: loss=0.0036470760, psnr=24.380552, ssim=0.908773, lpips=unavailable
  image_021: loss=0.0043273913, psnr=23.637738, ssim=0.907131, lpips=unavailable
  image_022: loss=0.0026796949, psnr=25.719146, ssim=0.927711, lpips=unavailable
  image_023: loss=0.0021203950, psnr=26.735832, ssim=0.932768, lpips=unavailable
  image_024: loss=0.0015859731, psnr=27.997042, ssim=0.940049, lpips=unavailable
