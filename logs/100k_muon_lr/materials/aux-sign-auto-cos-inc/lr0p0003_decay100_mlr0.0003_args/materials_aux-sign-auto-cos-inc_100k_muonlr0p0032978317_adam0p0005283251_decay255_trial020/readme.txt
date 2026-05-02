====================================================================================================
saved_time: 2026-05-01 21:16:46
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/materials.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/materials/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname materials_aux-sign-auto-cos-inc_100k_muonlr0p0032978317_adam0p0005283251_decay255_trial020 --optimizer aux-sign-auto-cos-inc --lrate 0.0005283251356203727 --lrate_decay 255 --muon_lrate 0.003297831667884783 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/materials/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/materials_aux-sign-auto-cos-inc_100k_muonlr0p0032978317_adam0p0005283251_decay255_trial020/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: materials_aux-sign-auto-cos-inc_100k_muonlr0p0032978317_adam0p0005283251_decay255_trial020
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 7 min
current_train_loss: 0.0035784564
current_train_psnr: 29.826090
testset_mean_loss: 0.0010554338
testset_mean_psnr: 29.949907
testset_mean_ssim: 0.962733
testset_mean_lpips: 0.022236
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0017342095, psnr=27.608984, ssim=0.951180, lpips=0.026506
  image_001: loss=0.0020571980, psnr=26.867239, ssim=0.945179, lpips=0.032541
  image_002: loss=0.0015308690, psnr=28.150619, ssim=0.955010, lpips=0.026643
  image_003: loss=0.0011099322, psnr=29.547035, ssim=0.963233, lpips=0.024533
  image_004: loss=0.0010152189, psnr=29.934403, ssim=0.966923, lpips=0.021376
  image_005: loss=0.0009843525, psnr=30.068493, ssim=0.962307, lpips=0.019638
  image_006: loss=0.0006236669, psnr=32.050473, ssim=0.969943, lpips=0.014832
  image_007: loss=0.0013088369, psnr=28.831145, ssim=0.944231, lpips=0.025831
  image_008: loss=0.0011596936, psnr=29.356567, ssim=0.950168, lpips=0.023305
  image_009: loss=0.0008714158, psnr=30.597745, ssim=0.966681, lpips=0.021173
  image_010: loss=0.0007857601, psnr=31.047100, ssim=0.965407, lpips=0.022429
  image_011: loss=0.0008618400, psnr=30.645733, ssim=0.969591, lpips=0.023185
  image_012: loss=0.0012128863, psnr=29.161799, ssim=0.964950, lpips=0.026356
  image_013: loss=0.0009200040, psnr=30.362102, ssim=0.972075, lpips=0.021450
  image_014: loss=0.0008571891, psnr=30.669233, ssim=0.972302, lpips=0.021903
  image_015: loss=0.0011776619, psnr=29.289793, ssim=0.959236, lpips=0.031963
  image_016: loss=0.0009034119, psnr=30.441141, ssim=0.971868, lpips=0.019366
  image_017: loss=0.0012213739, psnr=29.131513, ssim=0.957750, lpips=0.022182
  image_018: loss=0.0009485715, psnr=30.229299, ssim=0.960541, lpips=0.016497
  image_019: loss=0.0008293314, psnr=30.812718, ssim=0.961080, lpips=0.017055
  image_020: loss=0.0012195484, psnr=29.138009, ssim=0.951454, lpips=0.024356
  image_021: loss=0.0007713032, psnr=31.127748, ssim=0.970134, lpips=0.019484
  image_022: loss=0.0006244433, psnr=32.045069, ssim=0.973990, lpips=0.019003
  image_023: loss=0.0008347236, psnr=30.784572, ssim=0.970289, lpips=0.018068
  image_024: loss=0.0008224031, psnr=30.849152, ssim=0.972795, lpips=0.016227
