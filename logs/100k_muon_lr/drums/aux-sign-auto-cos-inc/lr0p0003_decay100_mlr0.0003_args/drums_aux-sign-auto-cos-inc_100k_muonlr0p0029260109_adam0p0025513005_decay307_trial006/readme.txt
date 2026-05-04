====================================================================================================
saved_time: 2026-05-03 06:54:08
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0029260109_adam0p0025513005_decay307_trial006 --optimizer aux-sign-auto-cos-inc --lrate 0.0025513004740846646 --lrate_decay 307 --muon_lrate 0.002926010906487721 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0029260109_adam0p0025513005_decay307_trial006/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0029260109_adam0p0025513005_decay307_trial006
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0048963460
current_train_psnr: 28.407673
testset_mean_loss: 0.0028197365
testset_mean_psnr: 25.764996
testset_mean_ssim: 0.932021
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0016709457, psnr=27.770376, ssim=0.934551, lpips=unavailable
  image_001: loss=0.0020694833, psnr=26.841381, ssim=0.929936, lpips=unavailable
  image_002: loss=0.0012936955, psnr=28.881679, ssim=0.950281, lpips=unavailable
  image_003: loss=0.0022954021, psnr=26.391412, ssim=0.939173, lpips=unavailable
  image_004: loss=0.0018814936, psnr=27.254972, ssim=0.937912, lpips=unavailable
  image_005: loss=0.0017252812, psnr=27.631401, ssim=0.943999, lpips=unavailable
  image_006: loss=0.0043816231, psnr=23.583650, ssim=0.906433, lpips=unavailable
  image_007: loss=0.0034094944, psnr=24.673100, ssim=0.920802, lpips=unavailable
  image_008: loss=0.0028519330, psnr=25.448607, ssim=0.935232, lpips=unavailable
  image_009: loss=0.0034524996, psnr=24.618663, ssim=0.933188, lpips=unavailable
  image_010: loss=0.0034535527, psnr=24.617339, ssim=0.937216, lpips=unavailable
  image_011: loss=0.0040933811, psnr=23.879178, ssim=0.920462, lpips=unavailable
  image_012: loss=0.0036171705, psnr=24.416310, ssim=0.930686, lpips=unavailable
  image_013: loss=0.0025715807, psnr=25.897998, ssim=0.939152, lpips=unavailable
  image_014: loss=0.0024354171, psnr=26.134266, ssim=0.941230, lpips=unavailable
  image_015: loss=0.0039524445, psnr=24.031342, ssim=0.934552, lpips=unavailable
  image_016: loss=0.0019843034, psnr=27.023919, ssim=0.951620, lpips=unavailable
  image_017: loss=0.0026566221, psnr=25.756702, ssim=0.935288, lpips=unavailable
  image_018: loss=0.0029618086, psnr=25.284430, ssim=0.931456, lpips=unavailable
  image_019: loss=0.0050130105, psnr=22.999014, ssim=0.890981, lpips=unavailable
  image_020: loss=0.0031688996, psnr=24.990915, ssim=0.920821, lpips=unavailable
  image_021: loss=0.0038066965, psnr=24.194517, ssim=0.916402, lpips=unavailable
  image_022: loss=0.0023904571, psnr=26.215190, ssim=0.934962, lpips=unavailable
  image_023: loss=0.0019116603, psnr=27.185893, ssim=0.939113, lpips=unavailable
  image_024: loss=0.0014445556, psnr=28.402657, ssim=0.945063, lpips=unavailable
