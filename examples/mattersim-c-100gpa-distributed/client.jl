using EDDPotentials

worker_id = get(ENV, "WORKER_ID", length(ARGS) >= 1 ? ARGS[1] : string(getpid()))
EDDPotentials.distributed_worker("link.toml"; worker_id=worker_id)
