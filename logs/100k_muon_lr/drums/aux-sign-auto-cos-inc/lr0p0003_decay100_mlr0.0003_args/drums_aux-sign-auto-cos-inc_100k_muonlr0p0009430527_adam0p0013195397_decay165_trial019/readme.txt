====================================================================================================
saved_time: 2026-05-04 11:52:14
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0009430527_adam0p0013195397_decay165_trial019 --optimizer aux-sign-auto-cos-inc --lrate 0.0013195397324116162 --lrate_decay 165 --muon_lrate 0.0009430526757093006 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0009430527_adam0p0013195397_decay165_trial019/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0009430527_adam0p0013195397_decay165_trial019
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0050639203
current_train_psnr: 27.628870
testset_mean_loss: 0.0030106564
testset_mean_psnr: 25.486491
testset_mean_ssim: 0.927917
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017955197, psnr=27.458098, ssim=0.929569, lpips=unavailable
  image_001: loss=0.0022063530, psnr=26.563250, ssim=0.925502, lpips=unavailable
  image_002: loss=0.0013929056, psnr=28.560783, ssim=0.947791, lpips=unavailable
  image_003: loss=0.0021762266, psnr=26.622959, ssim=0.938605, lpips=unavailable
  image_004: loss=0.0019809471, psnr=27.031271, ssim=0.933283, lpips=unavailable
  image_005: loss=0.0018782124, psnr=27.262553, ssim=0.939151, lpips=unavailable
  image_006: loss=0.0047923806, psnr=23.194487, ssim=0.902844, lpips=unavailable
  image_007: loss=0.0036937960, psnr=24.325271, ssim=0.916861, lpips=unavailable
  image_008: loss=0.0030517064, psnr=25.154572, ssim=0.931118, lpips=unavailable
  image_009: loss=0.0041468502, psnr=23.822816, ssim=0.926375, lpips=unavailable
  image_010: loss=0.0034093175, psnr=24.673325, ssim=0.934733, lpips=unavailable
  image_011: loss=0.0040225373, psnr=23.954999, ssim=0.915854, lpips=unavailable
  image_012: loss=0.0041281763, psnr=23.842418, ssim=0.923647, lpips=unavailable
  image_013: loss=0.0027412232, psnr=25.620556, ssim=0.934308, lpips=unavailable
  image_014: loss=0.0025242197, psnr=25.978728, ssim=0.938676, lpips=unavailable
  image_015: loss=0.0041391198, psnr=23.830920, ssim=0.929832, lpips=unavailable
  image_016: loss=0.0021995166, psnr=26.576727, ssim=0.948780, lpips=unavailable
  image_017: loss=0.0027576187, psnr=25.594658, ssim=0.931482, lpips=unavailable
  image_018: loss=0.0031703280, psnr=24.988958, ssim=0.927525, lpips=unavailable
  image_019: loss=0.0053805234, psnr=22.691755, ssim=0.891128, lpips=unavailable
  image_020: loss=0.0032945594, psnr=24.822026, ssim=0.915988, lpips=unavailable
  image_021: loss=0.0041615111, psnr=23.807489, ssim=0.909028, lpips=unavailable
  image_022: loss=0.0026047898, psnr=25.842273, ssim=0.930018, lpips=unavailable
  image_023: loss=0.0020685620, psnr=26.843314, ssim=0.934080, lpips=unavailable
  image_024: loss=0.0015495089, psnr=28.098059, ssim=0.941752, lpips=unavailable
