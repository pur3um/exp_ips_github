====================================================================================================
saved_time: 2026-04-30 11:29:05
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/ficus.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname ficus_aux-sign-auto-cos-inc_100k_muonlr0p0016353358_adam0p0010615361_decay386_trial000 --optimizer aux-sign-auto-cos-inc --lrate 0.0010615360772719654 --lrate_decay 386 --muon_lrate 0.001635335754309247 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/ficus/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/ficus_aux-sign-auto-cos-inc_100k_muonlr0p0016353358_adam0p0010615361_decay386_trial000/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: ficus_aux-sign-auto-cos-inc_100k_muonlr0p0016353358_adam0p0010615361_decay386_trial000
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 35 min
current_train_loss: 0.0023467676
current_train_psnr: 31.927795
testset_mean_loss: 0.0012287148
testset_mean_psnr: 29.259227
testset_mean_ssim: 0.966689
testset_mean_lpips: 0.023503
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0008749633, psnr=30.580101, ssim=0.969995, lpips=0.022949
  image_001: loss=0.0013529019, psnr=28.687337, ssim=0.965454, lpips=0.026577
  image_002: loss=0.0015633048, psnr=28.059563, ssim=0.960764, lpips=0.026885
  image_003: loss=0.0016981801, psnr=27.700162, ssim=0.956485, lpips=0.029669
  image_004: loss=0.0018081724, psnr=27.427601, ssim=0.957445, lpips=0.030793
  image_005: loss=0.0012312753, psnr=29.096448, ssim=0.967151, lpips=0.026619
  image_006: loss=0.0013064884, psnr=28.838944, ssim=0.966450, lpips=0.024228
  image_007: loss=0.0008438103, psnr=30.737551, ssim=0.974853, lpips=0.016688
  image_008: loss=0.0013244674, psnr=28.779587, ssim=0.960859, lpips=0.028411
  image_009: loss=0.0014817982, psnr=28.292109, ssim=0.958996, lpips=0.027168
  image_010: loss=0.0008944349, psnr=30.484512, ssim=0.974082, lpips=0.017747
  image_011: loss=0.0008203216, psnr=30.860158, ssim=0.978193, lpips=0.014244
  image_012: loss=0.0008240234, psnr=30.840604, ssim=0.977468, lpips=0.016365
  image_013: loss=0.0007217811, psnr=31.415944, ssim=0.979556, lpips=0.014838
  image_014: loss=0.0011412423, psnr=29.426221, ssim=0.969040, lpips=0.019430
  image_015: loss=0.0011320479, psnr=29.461352, ssim=0.964712, lpips=0.024377
  image_016: loss=0.0011301247, psnr=29.468736, ssim=0.973391, lpips=0.017017
  image_017: loss=0.0010532383, psnr=29.774733, ssim=0.972773, lpips=0.021523
  image_018: loss=0.0011449326, psnr=29.412200, ssim=0.969004, lpips=0.019819
  image_019: loss=0.0012982695, psnr=28.866351, ssim=0.966691, lpips=0.022796
  image_020: loss=0.0015983435, psnr=27.963299, ssim=0.959717, lpips=0.030178
  image_021: loss=0.0008052075, psnr=30.940921, ssim=0.970176, lpips=0.021637
  image_022: loss=0.0013476186, psnr=28.704330, ssim=0.959913, lpips=0.026645
  image_023: loss=0.0018651056, psnr=27.292966, ssim=0.955018, lpips=0.034915
  image_024: loss=0.0014558154, psnr=28.368937, ssim=0.959046, lpips=0.026053
