====================================================================================================
saved_time: 2026-05-03 14:51:48
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/mic.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0020441557_adam0p0015827249_decay111_trial015 --optimizer aux-sign-auto-cos-inc --lrate 0.0015827249451338603 --lrate_decay 111 --muon_lrate 0.0020441557254476405 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0020441557_adam0p0015827249_decay111_trial015/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0020441557_adam0p0015827249_decay111_trial015
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0013358002
current_train_psnr: 34.834152
testset_mean_loss: 0.0004476385
testset_mean_psnr: 33.682367
testset_mean_ssim: 0.980281
testset_mean_lpips: 0.021753
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004532724, psnr=33.436406, ssim=0.979407, lpips=0.014489
  image_001: loss=0.0002820881, psnr=35.496151, ssim=0.984464, lpips=0.019000
  image_002: loss=0.0005005086, psnr=33.005883, ssim=0.978272, lpips=0.024110
  image_003: loss=0.0004066148, psnr=33.908167, ssim=0.977897, lpips=0.024054
  image_004: loss=0.0004136866, psnr=33.833285, ssim=0.975885, lpips=0.025258
  image_005: loss=0.0003619031, psnr=34.414076, ssim=0.980069, lpips=0.027008
  image_006: loss=0.0003124402, psnr=35.052329, ssim=0.991384, lpips=0.010934
  image_007: loss=0.0004050895, psnr=33.924489, ssim=0.983638, lpips=0.012488
  image_008: loss=0.0005213948, psnr=32.828332, ssim=0.971490, lpips=0.035718
  image_009: loss=0.0004273644, psnr=33.692016, ssim=0.975874, lpips=0.021579
  image_010: loss=0.0002819435, psnr=35.498377, ssim=0.983298, lpips=0.013833
  image_011: loss=0.0002586773, psnr=35.872414, ssim=0.986148, lpips=0.016340
  image_012: loss=0.0010361044, psnr=29.845964, ssim=0.977880, lpips=0.020472
  image_013: loss=0.0006043818, psnr=32.186885, ssim=0.983244, lpips=0.014439
  image_014: loss=0.0005098119, psnr=32.925899, ssim=0.983036, lpips=0.015898
  image_015: loss=0.0004431205, psnr=33.534781, ssim=0.981647, lpips=0.018632
  image_016: loss=0.0003802047, psnr=34.199824, ssim=0.978943, lpips=0.018044
  image_017: loss=0.0004135889, psnr=33.834311, ssim=0.976928, lpips=0.026055
  image_018: loss=0.0003138115, psnr=35.033310, ssim=0.985669, lpips=0.012727
  image_019: loss=0.0004786119, psnr=33.200165, ssim=0.989471, lpips=0.012215
  image_020: loss=0.0005135583, psnr=32.894101, ssim=0.974319, lpips=0.037303
  image_021: loss=0.0005137399, psnr=32.892566, ssim=0.973054, lpips=0.040448
  image_022: loss=0.0005151520, psnr=32.880645, ssim=0.974935, lpips=0.031304
  image_023: loss=0.0005055536, psnr=32.962327, ssim=0.977523, lpips=0.028758
  image_024: loss=0.0003383390, psnr=34.706478, ssim=0.982558, lpips=0.022710
