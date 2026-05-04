====================================================================================================
saved_time: 2026-05-02 17:48:14
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0012019451_adam0p0010615361_decay386_trial000 --optimizer aux-sign-auto-cos-inc --lrate 0.0010615360772719654 --lrate_decay 386 --muon_lrate 0.0012019450985893186 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0012019451_adam0p0010615361_decay386_trial000/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0012019451_adam0p0010615361_decay386_trial000
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 9 min
current_train_loss: 0.0053232345
current_train_psnr: 28.519226
testset_mean_loss: 0.0029535406
testset_mean_psnr: 25.562150
testset_mean_ssim: 0.929157
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017641721, psnr=27.534590, ssim=0.930280, lpips=unavailable
  image_001: loss=0.0021452841, psnr=26.685152, ssim=0.926715, lpips=unavailable
  image_002: loss=0.0013618646, psnr=28.658660, ssim=0.948898, lpips=unavailable
  image_003: loss=0.0023250747, psnr=26.335631, ssim=0.936582, lpips=unavailable
  image_004: loss=0.0019946364, psnr=27.001362, ssim=0.935459, lpips=unavailable
  image_005: loss=0.0017928504, psnr=27.464559, ssim=0.941247, lpips=unavailable
  image_006: loss=0.0044948612, psnr=23.472837, ssim=0.904486, lpips=unavailable
  image_007: loss=0.0036242930, psnr=24.407767, ssim=0.917297, lpips=unavailable
  image_008: loss=0.0029757577, psnr=25.264024, ssim=0.932469, lpips=unavailable
  image_009: loss=0.0038182777, psnr=24.181325, ssim=0.929313, lpips=unavailable
  image_010: loss=0.0034272033, psnr=24.650601, ssim=0.935528, lpips=unavailable
  image_011: loss=0.0040655029, psnr=23.908857, ssim=0.918261, lpips=unavailable
  image_012: loss=0.0041100862, psnr=23.861491, ssim=0.926009, lpips=unavailable
  image_013: loss=0.0027027149, psnr=25.681998, ssim=0.935922, lpips=unavailable
  image_014: loss=0.0025359974, psnr=25.958512, ssim=0.938849, lpips=unavailable
  image_015: loss=0.0042350991, psnr=23.731364, ssim=0.932323, lpips=unavailable
  image_016: loss=0.0020967792, psnr=26.784473, ssim=0.951276, lpips=unavailable
  image_017: loss=0.0027030194, psnr=25.681508, ssim=0.932816, lpips=unavailable
  image_018: loss=0.0030123158, psnr=25.210995, ssim=0.929589, lpips=unavailable
  image_019: loss=0.0051678484, psnr=22.866902, ssim=0.889858, lpips=unavailable
  image_020: loss=0.0032859407, psnr=24.833403, ssim=0.915884, lpips=unavailable
  image_021: loss=0.0040347436, psnr=23.941840, ssim=0.912525, lpips=unavailable
  image_022: loss=0.0026121114, psnr=25.830083, ssim=0.930117, lpips=unavailable
  image_023: loss=0.0020371203, psnr=26.909833, ssim=0.934463, lpips=unavailable
  image_024: loss=0.0015149597, psnr=28.195989, ssim=0.942768, lpips=unavailable
