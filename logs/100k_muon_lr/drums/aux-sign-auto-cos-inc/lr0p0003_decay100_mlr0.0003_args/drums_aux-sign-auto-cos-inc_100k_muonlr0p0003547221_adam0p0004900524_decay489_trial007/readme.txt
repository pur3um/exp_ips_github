====================================================================================================
saved_time: 2026-05-03 09:08:15
script_path: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py
executed_command: /workspace/exp_ips_github/run_ranksched_optims_optuna_ready.py --config /workspace/exp_ips_github/configs/drums.txt --basedir /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args --expname drums_aux-sign-auto-cos-inc_100k_muonlr0p0003547221_adam0p0004900524_decay489_trial007 --optimizer aux-sign-auto-cos-inc --lrate 0.0004900524124859565 --lrate_decay 489 --muon_lrate 0.00035472207152279894 --muon_decay 0.0 --muon_momentum 0.9 --N_iters 100000 --eval_every 5000 --max_eval_views 2 --metric_out /workspace/exp_ips_github/logs/100k_muon_lr/drums/aux-sign-auto-cos-inc/lr0p0003_decay100_mlr0.0003_args/drums_aux-sign-auto-cos-inc_100k_muonlr0p0003547221_adam0p0004900524_decay489_trial007/metric.json --no_reload --seed 0 --lowrank_auto_init_rank_start
expname: drums_aux-sign-auto-cos-inc_100k_muonlr0p0003547221_adam0p0004900524_decay489_trial007
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 13 min
current_train_loss: 0.0060699265
current_train_psnr: 26.601856
testset_mean_loss: 0.0035608320
testset_mean_psnr: 24.677774
testset_mean_ssim: 0.915779
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0022641986, psnr=26.450855, ssim=0.918399, lpips=unavailable
  image_001: loss=0.0026845962, psnr=25.711210, ssim=0.914354, lpips=unavailable
  image_002: loss=0.0020393871, psnr=26.905003, ssim=0.932385, lpips=unavailable
  image_003: loss=0.0025606626, psnr=25.916476, ssim=0.931369, lpips=unavailable
  image_004: loss=0.0024321126, psnr=26.140163, ssim=0.925142, lpips=unavailable
  image_005: loss=0.0024256345, psnr=26.151746, ssim=0.926522, lpips=unavailable
  image_006: loss=0.0048499228, psnr=23.142652, ssim=0.896722, lpips=unavailable
  image_007: loss=0.0043656235, psnr=23.599537, ssim=0.900055, lpips=unavailable
  image_008: loss=0.0039954116, psnr=23.984385, ssim=0.910763, lpips=unavailable
  image_009: loss=0.0046051950, psnr=23.367520, ssim=0.914813, lpips=unavailable
  image_010: loss=0.0044165705, psnr=23.549148, ssim=0.915963, lpips=unavailable
  image_011: loss=0.0048288293, psnr=23.161581, ssim=0.901321, lpips=unavailable
  image_012: loss=0.0044936780, psnr=23.473980, ssim=0.915019, lpips=unavailable
  image_013: loss=0.0035459944, psnr=24.502619, ssim=0.919093, lpips=unavailable
  image_014: loss=0.0032135991, psnr=24.930083, ssim=0.923069, lpips=unavailable
  image_015: loss=0.0046553290, psnr=23.320496, ssim=0.921967, lpips=unavailable
  image_016: loss=0.0029377821, psnr=25.319804, ssim=0.934506, lpips=unavailable
  image_017: loss=0.0033707810, psnr=24.722695, ssim=0.914205, lpips=unavailable
  image_018: loss=0.0040472480, psnr=23.928402, ssim=0.909633, lpips=unavailable
  image_019: loss=0.0055478527, psnr=22.558751, ssim=0.886375, lpips=unavailable
  image_020: loss=0.0041607819, psnr=23.808250, ssim=0.898567, lpips=unavailable
  image_021: loss=0.0043109050, psnr=23.654315, ssim=0.902888, lpips=unavailable
  image_022: loss=0.0028998200, psnr=25.376289, ssim=0.922028, lpips=unavailable
  image_023: loss=0.0024299591, psnr=26.144010, ssim=0.926275, lpips=unavailable
  image_024: loss=0.0019389263, psnr=27.124387, ssim=0.933049, lpips=unavailable
