====================================================================================================
saved_time: 2026-05-02 07:32:31
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0028173375_adam0p0005105011_decay494_trial011 --optimizer aux-sign-auto-cos-inc --lrate 0.0005105010993840165 --lrate_decay 494 --muon_lrate 0.002817337468089827 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0028173375_adam0p0005105011_decay494_trial011/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0028173375_adam0p0005105011_decay494_trial011
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 29 min
current_train_loss: 0.0009931190
current_train_psnr: 36.078953
testset_mean_loss: 0.0002262833
testset_mean_psnr: 37.356881
testset_mean_ssim: 0.981980
testset_mean_lpips: 0.014390
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001860549, psnr=37.303587, ssim=0.982707, lpips=0.014332
  image_001: loss=0.0001741743, psnr=37.590157, ssim=0.984311, lpips=0.015044
  image_002: loss=0.0002113530, psnr=36.749915, ssim=0.980254, lpips=0.018615
  image_003: loss=0.0001435637, psnr=38.429550, ssim=0.983211, lpips=0.016782
  image_004: loss=0.0001081849, psnr=39.658328, ssim=0.985608, lpips=0.012131
  image_005: loss=0.0001531164, psnr=38.149780, ssim=0.984245, lpips=0.010652
  image_006: loss=0.0001987714, psnr=37.016459, ssim=0.983792, lpips=0.011225
  image_007: loss=0.0001313295, psnr=38.816374, ssim=0.986452, lpips=0.009733
  image_008: loss=0.0001405748, psnr=38.520923, ssim=0.982001, lpips=0.011882
  image_009: loss=0.0001546462, psnr=38.106605, ssim=0.977815, lpips=0.016314
  image_010: loss=0.0001351484, psnr=38.691888, ssim=0.979378, lpips=0.016694
  image_011: loss=0.0002249691, psnr=36.478769, ssim=0.976742, lpips=0.018662
  image_012: loss=0.0002293742, psnr=36.394553, ssim=0.984522, lpips=0.009930
  image_013: loss=0.0001459041, psnr=38.359321, ssim=0.989340, lpips=0.007515
  image_014: loss=0.0002763775, psnr=35.584971, ssim=0.983499, lpips=0.013823
  image_015: loss=0.0012742481, psnr=28.947460, ssim=0.964308, lpips=0.033026
  image_016: loss=0.0005638887, psnr=32.488065, ssim=0.970147, lpips=0.026659
  image_017: loss=0.0001381707, psnr=38.595837, ssim=0.983598, lpips=0.014137
  image_018: loss=0.0002095255, psnr=36.787629, ssim=0.981847, lpips=0.011022
  image_019: loss=0.0001742793, psnr=37.587538, ssim=0.984931, lpips=0.010103
  image_020: loss=0.0001153633, psnr=39.379319, ssim=0.987105, lpips=0.008842
  image_021: loss=0.0001148456, psnr=39.398853, ssim=0.984583, lpips=0.010052
  image_022: loss=0.0001221099, psnr=39.132488, ssim=0.983436, lpips=0.015019
  image_023: loss=0.0001368621, psnr=38.637165, ssim=0.983399, lpips=0.013813
  image_024: loss=0.0001942458, psnr=37.116482, ssim=0.982272, lpips=0.013755
