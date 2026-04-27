# synth: chair, drums, ficus, hotdog, lego, materials, mic, ship
# flower, fern, fortress, horns, leaves, orchids, room, trex

CUDA_VISIBLE_DEVICES=0 python run_nerf_ranksched.py --basedir logs_104/sched/cosine --config configs/ficus.txt --expname ficus_auto_200k --optimizer aux-sign-auto-cos-inc --train_scheduler warmup_cosine --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 200000&
CUDA_VISIBLE_DEVICES=1 python run_nerf_ranksched.py --basedir logs_104/sched/rankwsd --config configs/ficus.txt --expname ficus_auto_200k --optimizer aux-sign-auto-cos-inc --train_scheduler rank_wsd --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 200000&
wait
CUDA_VISIBLE_DEVICES=0 python run_nerf_ranksched.py --basedir logs_104/sched/cosine --config configs/hotdog.txt --expname hotdog_auto_200k --optimizer aux-sign-auto-cos-inc --train_scheduler warmup_cosine --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 200000&
CUDA_VISIBLE_DEVICES=1 python run_nerf_ranksched.py --basedir logs_104/sched/rankwsd --config configs/hotdog.txt --expname hotdog_auto_200k --optimizer aux-sign-auto-cos-inc --train_scheduler rank_wsd --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 200000&
wait
CUDA_VISIBLE_DEVICES=0 python run_nerf_ranksched.py --basedir logs_104/sched/cosine --config configs/ship.txt --expname ship_auto_200k --optimizer aux-sign-auto-cos-inc --train_scheduler warmup_cosine --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 200000&
CUDA_VISIBLE_DEVICES=1 python run_nerf_ranksched.py --basedir logs_104/sched/rankwsd --config configs/ship.txt --expname ship_auto_200k --optimizer aux-sign-auto-cos-inc --train_scheduler rank_wsd --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 200000&
wait
# drum, fern, flower,
# CUDA_VISIBLE_DEVICES=5 python run_nerf_ranksched.py --basedir logs/sched/rankwsd --config configs/lego.txt --expname lego_auto_200k --optimizer aux-sign-auto-cos-inc --train_scheduler rank_wsd --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 200000

#@ time check
# CUDA_VISIBLE_DEVICES=0 python run_nerf_ranksched.py --basedir logs/sched/rankwsd --config configs/horns.txt --expname horns_auto_200k --optimizer aux-sign-auto-cos-inc --train_scheduler rank_wsd --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 200000