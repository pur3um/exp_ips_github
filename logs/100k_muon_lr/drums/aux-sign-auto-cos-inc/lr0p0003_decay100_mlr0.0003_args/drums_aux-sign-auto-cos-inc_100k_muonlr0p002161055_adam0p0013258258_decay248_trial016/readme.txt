====================================================================================================
saved_time: 2026-05-04 05:11:21
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p002161055_adam0p0013258258_decay248_trial016 --optimizer aux-sign-auto-cos-inc --lrate 0.0013258257875009593 --lrate_decay 248 --muon_lrate 0.002161055020092052 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p002161055_adam0p0013258258_decay248_trial016/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p002161055_adam0p0013258258_decay248_trial016
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0043252241
current_train_psnr: 28.546089
testset_mean_loss: 0.0028953923
testset_mean_psnr: 25.707104
testset_mean_ssim: 0.931257
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0016621096, psnr=27.793403, ssim=0.932909, lpips=unavailable
  image_001: loss=0.0021226569, psnr=26.731202, ssim=0.929315, lpips=unavailable
  image_002: loss=0.0012599714, psnr=28.996393, ssim=0.950950, lpips=unavailable
  image_003: loss=0.0021244411, psnr=26.727553, ssim=0.940892, lpips=unavailable
  image_004: loss=0.0019259222, psnr=27.153612, ssim=0.936474, lpips=unavailable
  image_005: loss=0.0017239015, psnr=27.634875, ssim=0.944157, lpips=unavailable
  image_006: loss=0.0050138170, psnr=22.998315, ssim=0.903368, lpips=unavailable
  image_007: loss=0.0035073941, psnr=24.550154, ssim=0.920865, lpips=unavailable
  image_008: loss=0.0027710248, psnr=25.573596, ssim=0.935825, lpips=unavailable
  image_009: loss=0.0037729088, psnr=24.233237, ssim=0.931189, lpips=unavailable
  image_010: loss=0.0032303198, psnr=24.907545, ssim=0.938993, lpips=unavailable
  image_011: loss=0.0038590278, psnr=24.135221, ssim=0.920065, lpips=unavailable
  image_012: loss=0.0039673648, psnr=24.014979, ssim=0.927186, lpips=unavailable
  image_013: loss=0.0024981417, psnr=26.023829, ssim=0.939268, lpips=unavailable
  image_014: loss=0.0024046628, psnr=26.189458, ssim=0.941908, lpips=unavailable
  image_015: loss=0.0040201894, psnr=23.957535, ssim=0.932872, lpips=unavailable
  image_016: loss=0.0019251092, psnr=27.155446, ssim=0.954006, lpips=unavailable
  image_017: loss=0.0026249492, psnr=25.808791, ssim=0.934956, lpips=unavailable
  image_018: loss=0.0028949326, psnr=25.383615, ssim=0.933515, lpips=unavailable
  image_019: loss=0.0059954119, psnr=22.221810, ssim=0.887706, lpips=unavailable
  image_020: loss=0.0033113861, psnr=24.799902, ssim=0.917641, lpips=unavailable
  image_021: loss=0.0039696898, psnr=24.012434, ssim=0.911705, lpips=unavailable
  image_022: loss=0.0023947624, psnr=26.207376, ssim=0.933964, lpips=unavailable
  image_023: loss=0.0019438956, psnr=27.113270, ssim=0.937402, lpips=unavailable
  image_024: loss=0.0014608176, psnr=28.354040, ssim=0.944301, lpips=unavailable
