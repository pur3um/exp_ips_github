====================================================================================================
saved_time: 2026-05-03 04:40:37
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0026909121_adam0p0003143806_decay155_trial005 --optimizer aux-sign-auto-cos-inc --lrate 0.00031438059122156694 --lrate_decay 155 --muon_lrate 0.002690912124985378 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0026909121_adam0p0003143806_decay155_trial005/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0026909121_adam0p0003143806_decay155_trial005
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 10 min
current_train_loss: 0.0050781108
current_train_psnr: 27.883598
testset_mean_loss: 0.0028254273
testset_mean_psnr: 25.753794
testset_mean_ssim: 0.932201
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0016896548, psnr=27.722020, ssim=0.934629, lpips=unavailable
  image_001: loss=0.0020676388, psnr=26.845253, ssim=0.931032, lpips=unavailable
  image_002: loss=0.0013044928, psnr=28.845583, ssim=0.950536, lpips=unavailable
  image_003: loss=0.0021597417, psnr=26.655982, ssim=0.940360, lpips=unavailable
  image_004: loss=0.0018256392, psnr=27.385850, ssim=0.939260, lpips=unavailable
  image_005: loss=0.0017341942, psnr=27.609022, ssim=0.944790, lpips=unavailable
  image_006: loss=0.0041303514, psnr=23.840130, ssim=0.908759, lpips=unavailable
  image_007: loss=0.0034889716, psnr=24.573026, ssim=0.922011, lpips=unavailable
  image_008: loss=0.0029252158, psnr=25.338421, ssim=0.933906, lpips=unavailable
  image_009: loss=0.0035928718, psnr=24.445583, ssim=0.932386, lpips=unavailable
  image_010: loss=0.0032461965, psnr=24.886252, ssim=0.937794, lpips=unavailable
  image_011: loss=0.0039732833, psnr=24.008505, ssim=0.919864, lpips=unavailable
  image_012: loss=0.0037018429, psnr=24.315820, ssim=0.929567, lpips=unavailable
  image_013: loss=0.0025525473, psnr=25.930262, ssim=0.938943, lpips=unavailable
  image_014: loss=0.0024093322, psnr=26.181033, ssim=0.941402, lpips=unavailable
  image_015: loss=0.0040143891, psnr=23.963805, ssim=0.934137, lpips=unavailable
  image_016: loss=0.0019523287, psnr=27.094470, ssim=0.953312, lpips=unavailable
  image_017: loss=0.0027419026, psnr=25.619480, ssim=0.933925, lpips=unavailable
  image_018: loss=0.0030249872, psnr=25.192764, ssim=0.930999, lpips=unavailable
  image_019: loss=0.0050836126, psnr=22.938275, ssim=0.894962, lpips=unavailable
  image_020: loss=0.0032284472, psnr=24.910063, ssim=0.920880, lpips=unavailable
  image_021: loss=0.0038488079, psnr=24.146738, ssim=0.915859, lpips=unavailable
  image_022: loss=0.0024636306, psnr=26.084244, ssim=0.932816, lpips=unavailable
  image_023: loss=0.0020154242, psnr=26.956335, ssim=0.937603, lpips=unavailable
  image_024: loss=0.0014601792, psnr=28.355938, ssim=0.945289, lpips=unavailable
