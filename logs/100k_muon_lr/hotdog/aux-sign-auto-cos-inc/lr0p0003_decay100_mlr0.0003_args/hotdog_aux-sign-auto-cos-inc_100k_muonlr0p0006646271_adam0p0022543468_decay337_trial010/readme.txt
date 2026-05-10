====================================================================================================
saved_time: 2026-05-02 04:02:34
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/hotdog.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0006646271_adam0p0022543468_decay337_trial010 --optimizer aux-sign-auto-cos-inc --lrate 0.002254346836116652 --lrate_decay 337 --muon_lrate 0.0006646271302500596 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/hotdog/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0006646271_adam0p0022543468_decay337_trial010/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: hotdog_aux-sign-auto-cos-inc_100k_muonlr0p0006646271_adam0p0022543468_decay337_trial010
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 30 min
current_train_loss: 0.0011123726
current_train_psnr: 34.854538
testset_mean_loss: 0.0002717779
testset_mean_psnr: 36.406072
testset_mean_ssim: 0.977826
testset_mean_lpips: 0.021540
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0002241827, psnr=36.493977, ssim=0.979089, lpips=0.019922
  image_001: loss=0.0002105248, psnr=36.766964, ssim=0.980662, lpips=0.021133
  image_002: loss=0.0002676756, psnr=35.723910, ssim=0.975864, lpips=0.024463
  image_003: loss=0.0001852426, psnr=37.322589, ssim=0.979419, lpips=0.023444
  image_004: loss=0.0001361795, psnr=38.658879, ssim=0.982468, lpips=0.018014
  image_005: loss=0.0001861166, psnr=37.302148, ssim=0.981197, lpips=0.016008
  image_006: loss=0.0002501580, psnr=36.017854, ssim=0.979553, lpips=0.016971
  image_007: loss=0.0001604872, psnr=37.945594, ssim=0.983191, lpips=0.015767
  image_008: loss=0.0001851696, psnr=37.324300, ssim=0.977029, lpips=0.017782
  image_009: loss=0.0002162768, psnr=36.649898, ssim=0.969595, lpips=0.030538
  image_010: loss=0.0001795179, psnr=37.458919, ssim=0.971348, lpips=0.030764
  image_011: loss=0.0002942866, psnr=35.312293, ssim=0.971634, lpips=0.028378
  image_012: loss=0.0003045994, psnr=35.162708, ssim=0.981227, lpips=0.014940
  image_013: loss=0.0001940940, psnr=37.119876, ssim=0.986654, lpips=0.011136
  image_014: loss=0.0003239981, psnr=34.894574, ssim=0.980857, lpips=0.019700
  image_015: loss=0.0014336368, psnr=28.435608, ssim=0.962538, lpips=0.032646
  image_016: loss=0.0005327404, psnr=32.734843, ssim=0.966772, lpips=0.029779
  image_017: loss=0.0001810388, psnr=37.422280, ssim=0.980460, lpips=0.018945
  image_018: loss=0.0002721369, psnr=35.652124, ssim=0.977131, lpips=0.016282
  image_019: loss=0.0002167461, psnr=36.640486, ssim=0.980645, lpips=0.015355
  image_020: loss=0.0001390191, psnr=38.569252, ssim=0.983862, lpips=0.015734
  image_021: loss=0.0001434571, psnr=38.432777, ssim=0.980055, lpips=0.019885
  image_022: loss=0.0001549717, psnr=38.097472, ssim=0.977751, lpips=0.028993
  image_023: loss=0.0001738442, psnr=37.598395, ssim=0.978277, lpips=0.028470
  image_024: loss=0.0002283458, psnr=36.414068, ssim=0.978381, lpips=0.023458
