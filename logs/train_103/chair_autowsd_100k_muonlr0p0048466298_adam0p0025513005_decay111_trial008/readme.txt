====================================================================================================
saved_time: 2026-05-02 04:57:55
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: run_ranksched_optims_optuna_ready.py --config configs/chair.txt --basedir logs/train_103 --expname chair_autowsd_100k_muonlr0p0048466298_adam0p0025513005_decay111_trial008 --optimizer aux-sign-auto-cos-inc --lrate 0.0025513004740846646 --lrate_decay 111 --muon_lrate 0.004846629789555252 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 10000 --max_eval_views 2 --metric_out logs/train_103/chair/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/chair_aux-sign-auto-cos-inc_100k_muonlr0p0048466298_adam0p0025513005_decay111_trial008/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: chair_autowsd_100k_muonlr0p0048466298_adam0p0025513005_decay111_trial008
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 24 min
current_train_loss: 0.0020528855
current_train_psnr: 31.537254
testset_mean_loss: 0.0004238876
testset_mean_psnr: 33.969781
testset_mean_ssim: 0.977517
testset_mean_lpips: 0.016136
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001433877, psnr=38.434877, ssim=0.993501, lpips=0.006441
  image_001: loss=0.0002040299, psnr=36.903060, ssim=0.989165, lpips=0.006598
  image_002: loss=0.0004112739, psnr=33.858688, ssim=0.980004, lpips=0.016843
  image_003: loss=0.0004529892, psnr=33.439121, ssim=0.977237, lpips=0.016654
  image_004: loss=0.0005058756, psnr=32.959562, ssim=0.974685, lpips=0.018829
  image_005: loss=0.0005392014, psnr=32.682489, ssim=0.973623, lpips=0.015866
  image_006: loss=0.0005489826, psnr=32.604413, ssim=0.970689, lpips=0.018221
  image_007: loss=0.0005899915, psnr=32.291542, ssim=0.967012, lpips=0.023146
  image_008: loss=0.0004390257, psnr=33.575100, ssim=0.973732, lpips=0.016178
  image_009: loss=0.0004756099, psnr=33.227490, ssim=0.973621, lpips=0.016384
  image_010: loss=0.0006430386, psnr=31.917629, ssim=0.971162, lpips=0.025627
  image_011: loss=0.0004228184, psnr=33.738460, ssim=0.978394, lpips=0.017969
  image_012: loss=0.0002905905, psnr=35.367184, ssim=0.983923, lpips=0.014482
  image_013: loss=0.0003351753, psnr=34.747279, ssim=0.981368, lpips=0.015817
  image_014: loss=0.0002876153, psnr=35.411879, ssim=0.982964, lpips=0.014274
  image_015: loss=0.0004416409, psnr=33.549306, ssim=0.977515, lpips=0.020409
  image_016: loss=0.0003719991, psnr=34.294580, ssim=0.980125, lpips=0.013803
  image_017: loss=0.0005212793, psnr=32.829294, ssim=0.974998, lpips=0.018036
  image_018: loss=0.0005658845, psnr=32.472722, ssim=0.972243, lpips=0.018761
  image_019: loss=0.0005558807, psnr=32.550183, ssim=0.969138, lpips=0.017158
  image_020: loss=0.0004835214, psnr=33.155842, ssim=0.971261, lpips=0.017919
  image_021: loss=0.0003867872, psnr=34.125278, ssim=0.975182, lpips=0.015581
  image_022: loss=0.0004170909, psnr=33.797692, ssim=0.975783, lpips=0.017982
  image_023: loss=0.0003554897, psnr=34.491729, ssim=0.981105, lpips=0.013817
  image_024: loss=0.0002080111, psnr=36.819133, ssim=0.989489, lpips=0.006611
