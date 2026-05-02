====================================================================================================
saved_time: 2026-05-01 08:23:57
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0032007451_adam0p0003141717_decay333_trial006 --optimizer aux-sign-auto-cos-inc --lrate 0.00031417172594439365 --lrate_decay 333 --muon_lrate 0.003200745117531504 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0032007451_adam0p0003141717_decay333_trial006/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0032007451_adam0p0003141717_decay333_trial006
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 27 min
current_train_loss: 0.0022575723
current_train_psnr: 33.047050
testset_mean_loss: 0.0012403053
testset_mean_psnr: 29.208464
testset_mean_ssim: 0.966437
testset_mean_lpips: 0.023190
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008439386, psnr=30.736891, ssim=0.971059, lpips=0.019880
  image_001: loss=0.0013292695, psnr=28.763869, ssim=0.965553, lpips=0.022957
  image_002: loss=0.0016035079, psnr=27.949289, ssim=0.960652, lpips=0.026903
  image_003: loss=0.0017512826, psnr=27.566437, ssim=0.956764, lpips=0.028154
  image_004: loss=0.0017841617, psnr=27.485658, ssim=0.957840, lpips=0.039079
  image_005: loss=0.0012806939, psnr=28.925546, ssim=0.965757, lpips=0.024338
  image_006: loss=0.0012698149, psnr=28.962595, ssim=0.966120, lpips=0.024009
  image_007: loss=0.0008830680, psnr=30.540058, ssim=0.973818, lpips=0.014998
  image_008: loss=0.0013033336, psnr=28.849444, ssim=0.961749, lpips=0.026362
  image_009: loss=0.0014809218, psnr=28.294679, ssim=0.959084, lpips=0.026577
  image_010: loss=0.0009337686, psnr=30.297607, ssim=0.973340, lpips=0.017934
  image_011: loss=0.0008347653, psnr=30.784356, ssim=0.978041, lpips=0.013165
  image_012: loss=0.0008343930, psnr=30.786293, ssim=0.977394, lpips=0.015931
  image_013: loss=0.0007706176, psnr=31.131610, ssim=0.978238, lpips=0.015412
  image_014: loss=0.0011880504, psnr=29.251651, ssim=0.968808, lpips=0.020655
  image_015: loss=0.0011527203, psnr=29.382760, ssim=0.964902, lpips=0.024035
  image_016: loss=0.0011407168, psnr=29.428221, ssim=0.973175, lpips=0.017382
  image_017: loss=0.0010958393, psnr=29.602531, ssim=0.971654, lpips=0.020300
  image_018: loss=0.0011741462, psnr=29.302778, ssim=0.968672, lpips=0.021149
  image_019: loss=0.0012699574, psnr=28.962108, ssim=0.966435, lpips=0.021940
  image_020: loss=0.0013824656, psnr=28.593456, ssim=0.961874, lpips=0.028204
  image_021: loss=0.0008611798, psnr=30.649061, ssim=0.969241, lpips=0.023335
  image_022: loss=0.0013616572, psnr=28.659322, ssim=0.959013, lpips=0.026423
  image_023: loss=0.0020133753, psnr=26.960752, ssim=0.953030, lpips=0.033446
  image_024: loss=0.0014639864, psnr=28.344629, ssim=0.958703, lpips=0.027175
