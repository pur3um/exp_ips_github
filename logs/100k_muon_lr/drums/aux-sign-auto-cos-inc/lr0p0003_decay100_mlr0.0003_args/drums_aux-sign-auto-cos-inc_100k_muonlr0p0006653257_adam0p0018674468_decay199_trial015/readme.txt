====================================================================================================
saved_time: 2026-05-04 02:57:17
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0006653257_adam0p0018674468_decay199_trial015 --optimizer aux-sign-auto-cos-inc --lrate 0.0018674468257162972 --lrate_decay 199 --muon_lrate 0.0006653257145082276 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0006653257_adam0p0018674468_decay199_trial015/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0006653257_adam0p0018674468_decay199_trial015
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0050998824
current_train_psnr: 27.801226
testset_mean_loss: 0.0030980075
testset_mean_psnr: 25.331467
testset_mean_ssim: 0.925268
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0018611947, psnr=27.302082, ssim=0.925769, lpips=unavailable
  image_001: loss=0.0023651614, psnr=26.261392, ssim=0.922255, lpips=unavailable
  image_002: loss=0.0015020891, psnr=28.233043, ssim=0.944932, lpips=unavailable
  image_003: loss=0.0023397547, psnr=26.308296, ssim=0.935754, lpips=unavailable
  image_004: loss=0.0020882706, psnr=26.802132, ssim=0.932251, lpips=unavailable
  image_005: loss=0.0019285798, psnr=27.147624, ssim=0.938014, lpips=unavailable
  image_006: loss=0.0043952921, psnr=23.570122, ssim=0.901678, lpips=unavailable
  image_007: loss=0.0037507613, psnr=24.258806, ssim=0.910983, lpips=unavailable
  image_008: loss=0.0032429206, psnr=24.890637, ssim=0.926347, lpips=unavailable
  image_009: loss=0.0040989676, psnr=23.873255, ssim=0.925145, lpips=unavailable
  image_010: loss=0.0035741022, psnr=24.468330, ssim=0.931427, lpips=unavailable
  image_011: loss=0.0043847039, psnr=23.580597, ssim=0.911609, lpips=unavailable
  image_012: loss=0.0042728432, psnr=23.692830, ssim=0.922850, lpips=unavailable
  image_013: loss=0.0028296299, psnr=25.482703, ssim=0.933519, lpips=unavailable
  image_014: loss=0.0026407300, psnr=25.782760, ssim=0.935320, lpips=unavailable
  image_015: loss=0.0041872677, psnr=23.780693, ssim=0.930682, lpips=unavailable
  image_016: loss=0.0022482027, psnr=26.481645, ssim=0.946740, lpips=unavailable
  image_017: loss=0.0031172936, psnr=25.062223, ssim=0.926972, lpips=unavailable
  image_018: loss=0.0032953175, psnr=24.821027, ssim=0.923207, lpips=unavailable
  image_019: loss=0.0051413006, psnr=22.889270, ssim=0.889076, lpips=unavailable
  image_020: loss=0.0035902080, psnr=24.448804, ssim=0.908921, lpips=unavailable
  image_021: loss=0.0041641761, psnr=23.804709, ssim=0.907557, lpips=unavailable
  image_022: loss=0.0026990627, psnr=25.687870, ssim=0.928385, lpips=unavailable
  image_023: loss=0.0021102086, psnr=26.756746, ssim=0.932170, lpips=unavailable
  image_024: loss=0.0016221498, psnr=27.899090, ssim=0.940132, lpips=unavailable
