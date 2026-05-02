====================================================================================================
saved_time: 2026-04-30 19:06:08
script_path: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /home/greenx9/nerf-pytorch/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /home/greenx9/nerf-pytorch/exp_ips_github/configs/mic.txt --basedir /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname mic_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial005 --optimizer aux-sign-auto-cos-inc --lrate 0.0007253753172998836 --lrate_decay 417 --muon_lrate 0.001328467245876384 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /home/greenx9/nerf-pytorch/exp_ips_github/logs/100k_muon_lr/mic/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/mic_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial005/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: mic_aux-sign-auto-cos-inc_100k_muonlr0p0013284672_adam0p0007253753_decay417_trial005
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 35 min
current_train_loss: 0.0015815299
current_train_psnr: 33.349670
testset_mean_loss: 0.0004767208
testset_mean_psnr: 33.417734
testset_mean_ssim: 0.979298
testset_mean_lpips: 0.023437
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0004513416, psnr=33.454945, ssim=0.978668, lpips=0.015234
  image_001: loss=0.0002824280, psnr=35.490921, ssim=0.984061, lpips=0.019931
  image_002: loss=0.0005390894, psnr=32.683391, ssim=0.976289, lpips=0.025328
  image_003: loss=0.0004383779, psnr=33.581512, ssim=0.977005, lpips=0.027064
  image_004: loss=0.0004374364, psnr=33.590850, ssim=0.974672, lpips=0.027131
  image_005: loss=0.0003810229, psnr=34.190488, ssim=0.979448, lpips=0.026979
  image_006: loss=0.0003504679, psnr=34.553516, ssim=0.989988, lpips=0.013350
  image_007: loss=0.0004439733, psnr=33.526430, ssim=0.982058, lpips=0.015958
  image_008: loss=0.0005444997, psnr=32.640023, ssim=0.970807, lpips=0.038385
  image_009: loss=0.0004401706, psnr=33.563789, ssim=0.975361, lpips=0.024783
  image_010: loss=0.0003012035, psnr=35.211399, ssim=0.982168, lpips=0.016471
  image_011: loss=0.0002757262, psnr=35.595218, ssim=0.985561, lpips=0.017626
  image_012: loss=0.0011285988, psnr=29.474604, ssim=0.976938, lpips=0.022282
  image_013: loss=0.0006402693, psnr=31.936373, ssim=0.982970, lpips=0.014650
  image_014: loss=0.0005016580, psnr=32.995922, ssim=0.982560, lpips=0.016390
  image_015: loss=0.0004582368, psnr=33.389100, ssim=0.981045, lpips=0.017937
  image_016: loss=0.0003949967, psnr=34.034065, ssim=0.978442, lpips=0.020171
  image_017: loss=0.0004878496, psnr=33.117139, ssim=0.975785, lpips=0.026579
  image_018: loss=0.0003274879, psnr=34.848046, ssim=0.984620, lpips=0.015307
  image_019: loss=0.0005208545, psnr=32.832835, ssim=0.987751, lpips=0.012787
  image_020: loss=0.0005810749, psnr=32.357678, ssim=0.972955, lpips=0.040555
  image_021: loss=0.0005301904, psnr=32.755680, ssim=0.972517, lpips=0.042113
  image_022: loss=0.0005423099, psnr=32.657524, ssim=0.973385, lpips=0.032997
  image_023: loss=0.0005580889, psnr=32.532966, ssim=0.975714, lpips=0.031920
  image_024: loss=0.0003606665, psnr=34.428941, ssim=0.981676, lpips=0.024003
