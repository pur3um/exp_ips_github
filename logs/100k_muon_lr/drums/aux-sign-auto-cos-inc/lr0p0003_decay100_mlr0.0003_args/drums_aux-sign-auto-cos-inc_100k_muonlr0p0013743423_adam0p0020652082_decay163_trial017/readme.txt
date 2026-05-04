====================================================================================================
saved_time: 2026-05-04 07:24:57
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0013743423_adam0p0020652082_decay163_trial017 --optimizer aux-sign-auto-cos-inc --lrate 0.002065208206300762 --lrate_decay 163 --muon_lrate 0.0013743422940680234 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0013743423_adam0p0020652082_decay163_trial017/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0013743423_adam0p0020652082_decay163_trial017
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0044676531
current_train_psnr: 28.747236
testset_mean_loss: 0.0029722018
testset_mean_psnr: 25.574773
testset_mean_ssim: 0.929594
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017293633, psnr=27.621137, ssim=0.930985, lpips=unavailable
  image_001: loss=0.0021873992, psnr=26.600719, ssim=0.927895, lpips=unavailable
  image_002: loss=0.0012881517, psnr=28.900329, ssim=0.950243, lpips=unavailable
  image_003: loss=0.0021593261, psnr=26.656817, ssim=0.938618, lpips=unavailable
  image_004: loss=0.0019670024, psnr=27.061951, ssim=0.936474, lpips=unavailable
  image_005: loss=0.0017928632, psnr=27.464528, ssim=0.941858, lpips=unavailable
  image_006: loss=0.0048756362, psnr=23.119687, ssim=0.901281, lpips=unavailable
  image_007: loss=0.0035982695, psnr=24.439063, ssim=0.916004, lpips=unavailable
  image_008: loss=0.0028473295, psnr=25.455623, ssim=0.935297, lpips=unavailable
  image_009: loss=0.0038200093, psnr=24.179356, ssim=0.930482, lpips=unavailable
  image_010: loss=0.0033303273, psnr=24.775131, ssim=0.936742, lpips=unavailable
  image_011: loss=0.0039744806, psnr=24.007196, ssim=0.918374, lpips=unavailable
  image_012: loss=0.0044454285, psnr=23.520864, ssim=0.922563, lpips=unavailable
  image_013: loss=0.0026402606, psnr=25.783532, ssim=0.937169, lpips=unavailable
  image_014: loss=0.0024888769, psnr=26.039966, ssim=0.939862, lpips=unavailable
  image_015: loss=0.0041217953, psnr=23.849136, ssim=0.934233, lpips=unavailable
  image_016: loss=0.0020396877, psnr=26.904363, ssim=0.952485, lpips=unavailable
  image_017: loss=0.0026512505, psnr=25.765492, ssim=0.934646, lpips=unavailable
  image_018: loss=0.0030724851, psnr=25.125102, ssim=0.929196, lpips=unavailable
  image_019: loss=0.0057704835, psnr=22.387878, ssim=0.887566, lpips=unavailable
  image_020: loss=0.0035009885, psnr=24.558093, ssim=0.915015, lpips=unavailable
  image_021: loss=0.0039110244, psnr=24.077095, ssim=0.912414, lpips=unavailable
  image_022: loss=0.0025270120, psnr=25.973927, ssim=0.931578, lpips=unavailable
  image_023: loss=0.0020823388, psnr=26.814486, ssim=0.934733, lpips=unavailable
  image_024: loss=0.0014832544, psnr=28.287843, ssim=0.944147, lpips=unavailable
