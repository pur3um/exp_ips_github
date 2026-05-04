====================================================================================================
saved_time: 2026-05-03 20:15:40
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p001737761_adam0p0015944806_decay200_trial012 --optimizer aux-sign-auto-cos-inc --lrate 0.0015944805974740246 --lrate_decay 200 --muon_lrate 0.0017377609738177513 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p001737761_adam0p0015944806_decay200_trial012/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p001737761_adam0p0015944806_decay200_trial012
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0046484717
current_train_psnr: 28.647457
testset_mean_loss: 0.0029198644
testset_mean_psnr: 25.644199
testset_mean_ssim: 0.930367
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017014736, psnr=27.691748, ssim=0.931337, lpips=unavailable
  image_001: loss=0.0020250424, psnr=26.935659, ssim=0.929002, lpips=unavailable
  image_002: loss=0.0012551561, psnr=29.013022, ssim=0.951372, lpips=unavailable
  image_003: loss=0.0020918781, psnr=26.794636, ssim=0.940179, lpips=unavailable
  image_004: loss=0.0019964585, psnr=26.997397, ssim=0.934686, lpips=unavailable
  image_005: loss=0.0017768777, psnr=27.503424, ssim=0.941601, lpips=unavailable
  image_006: loss=0.0044774204, psnr=23.489721, ssim=0.904341, lpips=unavailable
  image_007: loss=0.0038054879, psnr=24.195896, ssim=0.918303, lpips=unavailable
  image_008: loss=0.0028624865, psnr=25.432565, ssim=0.934527, lpips=unavailable
  image_009: loss=0.0038937756, psnr=24.096291, ssim=0.929655, lpips=unavailable
  image_010: loss=0.0032214923, psnr=24.919429, ssim=0.937278, lpips=unavailable
  image_011: loss=0.0040225103, psnr=23.955028, ssim=0.919080, lpips=unavailable
  image_012: loss=0.0040396117, psnr=23.936604, ssim=0.926212, lpips=unavailable
  image_013: loss=0.0026015865, psnr=25.847617, ssim=0.937780, lpips=unavailable
  image_014: loss=0.0024384086, psnr=26.128935, ssim=0.940764, lpips=unavailable
  image_015: loss=0.0039683809, psnr=24.013866, ssim=0.933690, lpips=unavailable
  image_016: loss=0.0020640471, psnr=26.852804, ssim=0.951105, lpips=unavailable
  image_017: loss=0.0028889736, psnr=25.392564, ssim=0.932765, lpips=unavailable
  image_018: loss=0.0029662645, psnr=25.277901, ssim=0.931655, lpips=unavailable
  image_019: loss=0.0056046429, psnr=22.514520, ssim=0.890878, lpips=unavailable
  image_020: loss=0.0033717833, psnr=24.721403, ssim=0.917401, lpips=unavailable
  image_021: loss=0.0040089278, psnr=23.969717, ssim=0.912042, lpips=unavailable
  image_022: loss=0.0024255884, psnr=26.151829, ssim=0.933259, lpips=unavailable
  image_023: loss=0.0020126684, psnr=26.962277, ssim=0.936264, lpips=unavailable
  image_024: loss=0.0014756676, psnr=28.310114, ssim=0.943998, lpips=unavailable
