====================================================================================================
saved_time: 2026-05-03 12:39:40
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/ship.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ship_aux-sign-auto-cos-inc_100k_muonlr0p0009285151_adam0p0003928303_decay362_trial014 --optimizer aux-sign-auto-cos-inc --lrate 0.00039283034346079566 --lrate_decay 362 --muon_lrate 0.0009285151080066723 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/ship/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ship_aux-sign-auto-cos-inc_100k_muonlr0p0009285151_adam0p0003928303_decay362_trial014/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ship_aux-sign-auto-cos-inc_100k_muonlr0p0009285151_adam0p0003928303_decay362_trial014
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 14 min
current_train_loss: 0.0029517931
current_train_psnr: 30.982189
testset_mean_loss: 0.0012658708
testset_mean_psnr: 29.192590
testset_mean_ssim: 0.866767
testset_mean_lpips: 0.101140
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0010950321, psnr=29.605731, ssim=0.813541, lpips=0.152160
  image_001: loss=0.0012714565, psnr=28.956985, ssim=0.817484, lpips=0.153402
  image_002: loss=0.0010568792, psnr=29.759746, ssim=0.842095, lpips=0.134364
  image_003: loss=0.0014012788, psnr=28.534754, ssim=0.801502, lpips=0.153154
  image_004: loss=0.0011691165, psnr=29.321422, ssim=0.825393, lpips=0.110756
  image_005: loss=0.0009553607, psnr=30.198326, ssim=0.865892, lpips=0.096892
  image_006: loss=0.0014287941, psnr=28.450303, ssim=0.860552, lpips=0.104221
  image_007: loss=0.0012123226, psnr=29.163818, ssim=0.874323, lpips=0.091262
  image_008: loss=0.0010176974, psnr=29.923813, ssim=0.887430, lpips=0.082194
  image_009: loss=0.0010130514, psnr=29.943685, ssim=0.895605, lpips=0.069952
  image_010: loss=0.0009105850, psnr=30.406794, ssim=0.909876, lpips=0.058319
  image_011: loss=0.0018294707, psnr=27.376745, ssim=0.880859, lpips=0.085436
  image_012: loss=0.0028260644, psnr=25.488179, ssim=0.865997, lpips=0.105280
  image_013: loss=0.0021163726, psnr=26.744079, ssim=0.883375, lpips=0.087928
  image_014: loss=0.0016305583, psnr=27.876636, ssim=0.916239, lpips=0.066672
  image_015: loss=0.0010793064, psnr=29.668552, ssim=0.934828, lpips=0.052018
  image_016: loss=0.0009824662, psnr=30.076824, ssim=0.916992, lpips=0.053196
  image_017: loss=0.0011259137, psnr=29.484949, ssim=0.897023, lpips=0.066142
  image_018: loss=0.0016580148, psnr=27.804116, ssim=0.865407, lpips=0.099254
  image_019: loss=0.0013130475, psnr=28.817195, ssim=0.864179, lpips=0.101615
  image_020: loss=0.0010309155, psnr=29.867769, ssim=0.874451, lpips=0.101444
  image_021: loss=0.0009425308, psnr=30.257044, ssim=0.859488, lpips=0.108590
  image_022: loss=0.0007744795, psnr=31.109900, ssim=0.858393, lpips=0.114224
  image_023: loss=0.0007727059, psnr=31.119857, ssim=0.844277, lpips=0.134314
  image_024: loss=0.0010333501, psnr=29.857525, ssim=0.813970, lpips=0.145722
