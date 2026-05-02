====================================================================================================
saved_time: 2026-04-30 21:52:55
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/chair.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/chair/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname chair_aux-sign-auto-cos-inc_100k_muonlr0p0003681557_adam0p0003439496_decay138_trial009 --optimizer aux-sign-auto-cos-inc --lrate 0.0003439496417731218 --lrate_decay 138 --muon_lrate 0.0003681556505036925 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/chair/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/chair_aux-sign-auto-cos-inc_100k_muonlr0p0003681557_adam0p0003439496_decay138_trial009/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: chair_aux-sign-auto-cos-inc_100k_muonlr0p0003681557_adam0p0003439496_decay138_trial009
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 9 min
current_train_loss: 0.0023416134
current_train_psnr: 30.485435
testset_mean_loss: 0.0006494196
testset_mean_psnr: 32.095290
testset_mean_ssim: 0.964281
testset_mean_lpips: 0.037813
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0002389689, psnr=36.216584, ssim=0.987270, lpips=0.014956
  image_001: loss=0.0003327673, psnr=34.778592, ssim=0.980537, lpips=0.016999
  image_002: loss=0.0006418198, psnr=31.925868, ssim=0.967571, lpips=0.032073
  image_003: loss=0.0006992747, psnr=31.553521, ssim=0.963785, lpips=0.037498
  image_004: loss=0.0007837000, psnr=31.058501, ssim=0.959567, lpips=0.045247
  image_005: loss=0.0008886447, psnr=30.512718, ssim=0.954735, lpips=0.048413
  image_006: loss=0.0008719110, psnr=30.595278, ssim=0.952452, lpips=0.051392
  image_007: loss=0.0008624749, psnr=30.642535, ssim=0.949745, lpips=0.051335
  image_008: loss=0.0006905086, psnr=31.608308, ssim=0.958884, lpips=0.045047
  image_009: loss=0.0007084435, psnr=31.496947, ssim=0.961054, lpips=0.035465
  image_010: loss=0.0008638555, psnr=30.635588, ssim=0.960028, lpips=0.042618
  image_011: loss=0.0005879452, psnr=32.306631, ssim=0.967555, lpips=0.038823
  image_012: loss=0.0004773471, psnr=33.211656, ssim=0.970792, lpips=0.037291
  image_013: loss=0.0005293282, psnr=32.762748, ssim=0.967452, lpips=0.037573
  image_014: loss=0.0004367292, psnr=33.597877, ssim=0.973129, lpips=0.031549
  image_015: loss=0.0006055316, psnr=32.178631, ssim=0.966775, lpips=0.036257
  image_016: loss=0.0005796108, psnr=32.368635, ssim=0.969376, lpips=0.032184
  image_017: loss=0.0008414028, psnr=30.749960, ssim=0.958666, lpips=0.040780
  image_018: loss=0.0008983166, psnr=30.465705, ssim=0.954174, lpips=0.048680
  image_019: loss=0.0008563864, psnr=30.673302, ssim=0.951045, lpips=0.049676
  image_020: loss=0.0007413366, psnr=31.299845, ssim=0.953959, lpips=0.048991
  image_021: loss=0.0006084626, psnr=32.157660, ssim=0.961345, lpips=0.042878
  image_022: loss=0.0006187105, psnr=32.085125, ssim=0.963974, lpips=0.035849
  image_023: loss=0.0005467971, psnr=32.621737, ssim=0.970921, lpips=0.027405
  image_024: loss=0.0003252151, psnr=34.878291, ssim=0.982236, lpips=0.016347
