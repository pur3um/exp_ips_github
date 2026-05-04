====================================================================================================
saved_time: 2026-05-03 15:48:23
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0016101959_adam0p0015552069_decay203_trial010 --optimizer aux-sign-auto-cos-inc --lrate 0.0015552069456077104 --lrate_decay 203 --muon_lrate 0.0016101959218125913 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0016101959_adam0p0015552069_decay203_trial010/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0016101959_adam0p0015552069_decay203_trial010
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0049520805
current_train_psnr: 28.257856
testset_mean_loss: 0.0029639553
testset_mean_psnr: 25.611753
testset_mean_ssim: 0.930383
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0016677331, psnr=27.778734, ssim=0.932824, lpips=unavailable
  image_001: loss=0.0020979505, psnr=26.782047, ssim=0.929735, lpips=unavailable
  image_002: loss=0.0012628834, psnr=28.986367, ssim=0.950741, lpips=unavailable
  image_003: loss=0.0022238481, psnr=26.528949, ssim=0.938620, lpips=unavailable
  image_004: loss=0.0018933498, psnr=27.227691, ssim=0.936859, lpips=unavailable
  image_005: loss=0.0017564101, psnr=27.553740, ssim=0.942739, lpips=unavailable
  image_006: loss=0.0055740182, psnr=22.538316, ssim=0.899318, lpips=unavailable
  image_007: loss=0.0037177489, psnr=24.297199, ssim=0.917428, lpips=unavailable
  image_008: loss=0.0027616210, psnr=25.588359, ssim=0.936550, lpips=unavailable
  image_009: loss=0.0037367116, psnr=24.275104, ssim=0.930685, lpips=unavailable
  image_010: loss=0.0033693544, psnr=24.724533, ssim=0.937346, lpips=unavailable
  image_011: loss=0.0041807811, psnr=23.787426, ssim=0.918444, lpips=unavailable
  image_012: loss=0.0038064967, psnr=24.194745, ssim=0.926203, lpips=unavailable
  image_013: loss=0.0025857626, psnr=25.874113, ssim=0.939002, lpips=unavailable
  image_014: loss=0.0023541835, psnr=26.281597, ssim=0.942289, lpips=unavailable
  image_015: loss=0.0040688640, psnr=23.905268, ssim=0.934281, lpips=unavailable
  image_016: loss=0.0019901351, psnr=27.011174, ssim=0.952090, lpips=unavailable
  image_017: loss=0.0029567839, psnr=25.291804, ssim=0.933040, lpips=unavailable
  image_018: loss=0.0029542334, psnr=25.295552, ssim=0.931688, lpips=unavailable
  image_019: loss=0.0059361099, psnr=22.264981, ssim=0.885444, lpips=unavailable
  image_020: loss=0.0033435214, psnr=24.757959, ssim=0.916811, lpips=unavailable
  image_021: loss=0.0038091142, psnr=24.191760, ssim=0.914690, lpips=unavailable
  image_022: loss=0.0025044207, psnr=26.012927, ssim=0.932342, lpips=unavailable
  image_023: loss=0.0020658518, psnr=26.849008, ssim=0.936372, lpips=unavailable
  image_024: loss=0.0014809956, psnr=28.294462, ssim=0.944038, lpips=unavailable
