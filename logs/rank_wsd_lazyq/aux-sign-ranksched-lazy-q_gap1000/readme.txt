====================================================================================================
saved_time: 2026-04-25 19:26:18
script_path: /workspace/exp_ips_github/run_nerf_ranksched_lazyq.py
executed_command: run_nerf_ranksched_lazyq.py --basedir logs/rank_wsd_lazyq --expname aux-sign-ranksched-lazy-q_gap1000 --config configs/lego.txt --optimizer aux-sign-ranksched-lazy-q --train_scheduler rank_wsd --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --lazy_q_update_gap 1000 --lazy_q_rank_refresh_stride 8 --lazy_q_use_b_ema true --lazy_q_b_ema_decay 0.9 --sched_warmup_frac 0.01
expname: aux-sign-ranksched-lazy-q_gap1000
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 3 hour 41 min
current_train_loss: 0.0029307231
current_train_psnr: 31.613173
testset_mean_loss: 0.0008971206
testset_mean_psnr: 30.716734
testset_mean_ssim: 0.956053
testset_mean_lpips: 0.029518
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0006805484, psnr=31.671409, ssim=0.960541, lpips=0.026691
  image_001: loss=0.0018555457, psnr=27.315283, ssim=0.947663, lpips=0.034694
  image_002: loss=0.0005093031, psnr=32.930236, ssim=0.966321, lpips=0.021856
  image_003: loss=0.0006174372, psnr=32.094071, ssim=0.964004, lpips=0.027499
  image_004: loss=0.0008989690, psnr=30.462553, ssim=0.957134, lpips=0.031476
  image_005: loss=0.0006649297, psnr=31.772242, ssim=0.955976, lpips=0.029969
  image_006: loss=0.0009323578, psnr=30.304173, ssim=0.951881, lpips=0.031402
  image_007: loss=0.0006228655, psnr=32.056056, ssim=0.950995, lpips=0.029179
  image_008: loss=0.0006668515, psnr=31.759708, ssim=0.963040, lpips=0.022911
  image_009: loss=0.0012796038, psnr=28.929244, ssim=0.959194, lpips=0.036312
  image_010: loss=0.0017902754, psnr=27.470801, ssim=0.943870, lpips=0.047324
  image_011: loss=0.0013331319, psnr=28.751268, ssim=0.947101, lpips=0.035402
  image_012: loss=0.0008971975, psnr=30.471119, ssim=0.958555, lpips=0.026929
  image_013: loss=0.0008716430, psnr=30.596613, ssim=0.957339, lpips=0.027045
  image_014: loss=0.0006786936, psnr=31.683262, ssim=0.957832, lpips=0.023749
  image_015: loss=0.0008136929, psnr=30.895394, ssim=0.958986, lpips=0.029362
  image_016: loss=0.0008719589, psnr=30.595039, ssim=0.969170, lpips=0.026670
  image_017: loss=0.0008408216, psnr=30.752961, ssim=0.959416, lpips=0.031912
  image_018: loss=0.0008198555, psnr=30.862626, ssim=0.953951, lpips=0.029765
  image_019: loss=0.0006921314, psnr=31.598114, ssim=0.947813, lpips=0.029285
  image_020: loss=0.0005287817, psnr=32.767235, ssim=0.954822, lpips=0.023951
  image_021: loss=0.0007044520, psnr=31.521485, ssim=0.957576, lpips=0.022763
  image_022: loss=0.0009905697, psnr=30.041149, ssim=0.953484, lpips=0.033303
  image_023: loss=0.0009894015, psnr=30.046274, ssim=0.952048, lpips=0.031871
  image_024: loss=0.0008769958, psnr=30.570024, ssim=0.952624, lpips=0.026619
