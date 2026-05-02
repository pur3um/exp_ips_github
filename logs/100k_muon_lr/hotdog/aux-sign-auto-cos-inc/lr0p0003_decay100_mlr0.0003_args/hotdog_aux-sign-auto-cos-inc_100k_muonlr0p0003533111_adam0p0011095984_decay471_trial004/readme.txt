====================================================================================================
saved_time: 2026-05-01 07:02:45
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0003533111_adam0p0011095984_decay471_trial004 --optimizer aux-sign-auto-cos-inc --lrate 0.0011095983846308643 --lrate_decay 471 --muon_lrate 0.00035331112522565144 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0003533111_adam0p0011095984_decay471_trial004/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0003533111_adam0p0011095984_decay471_trial004
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 30 min
current_train_loss: 0.0013780756
current_train_psnr: 34.133259
testset_mean_loss: 0.0003376719
testset_mean_psnr: 35.219116
testset_mean_ssim: 0.972998
testset_mean_lpips: 0.029599
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0003048829, psnr=35.158669, ssim=0.973313, lpips=0.028177
  image_001: loss=0.0002928696, psnr=35.333256, ssim=0.974908, lpips=0.030798
  image_002: loss=0.0003432310, psnr=34.644134, ssim=0.971063, lpips=0.032280
  image_003: loss=0.0002499571, psnr=36.021343, ssim=0.973789, lpips=0.032144
  image_004: loss=0.0001908165, psnr=37.193839, ssim=0.977205, lpips=0.027009
  image_005: loss=0.0002531524, psnr=35.966178, ssim=0.975663, lpips=0.024846
  image_006: loss=0.0003280387, psnr=34.840747, ssim=0.973911, lpips=0.024809
  image_007: loss=0.0002267673, psnr=36.444194, ssim=0.978304, lpips=0.021381
  image_008: loss=0.0002290381, psnr=36.400920, ssim=0.971907, lpips=0.024747
  image_009: loss=0.0002565997, psnr=35.907437, ssim=0.964593, lpips=0.037680
  image_010: loss=0.0002280258, psnr=36.420158, ssim=0.965423, lpips=0.041209
  image_011: loss=0.0004310800, psnr=33.654421, ssim=0.966000, lpips=0.039177
  image_012: loss=0.0004173989, psnr=33.794486, ssim=0.976921, lpips=0.021151
  image_013: loss=0.0003331210, psnr=34.773979, ssim=0.981851, lpips=0.017380
  image_014: loss=0.0004146457, psnr=33.823227, ssim=0.977404, lpips=0.026508
  image_015: loss=0.0013427692, psnr=28.719986, ssim=0.961391, lpips=0.037917
  image_016: loss=0.0006166213, psnr=32.099814, ssim=0.965991, lpips=0.034491
  image_017: loss=0.0002396408, psnr=36.204390, ssim=0.975296, lpips=0.024516
  image_018: loss=0.0003427987, psnr=34.649607, ssim=0.971892, lpips=0.024120
  image_019: loss=0.0002830586, psnr=35.481234, ssim=0.975213, lpips=0.022530
  image_020: loss=0.0001877395, psnr=37.264440, ssim=0.979672, lpips=0.023835
  image_021: loss=0.0001917514, psnr=37.172613, ssim=0.975363, lpips=0.027976
  image_022: loss=0.0001990861, psnr=37.009588, ssim=0.972521, lpips=0.039737
  image_023: loss=0.0002275899, psnr=36.428468, ssim=0.972913, lpips=0.037040
  image_024: loss=0.0003111165, psnr=35.070769, ssim=0.972448, lpips=0.038505
