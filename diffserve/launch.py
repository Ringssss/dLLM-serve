"""
DiffServe CLI Launch Entry Point.

Usage:
  # Start HTTP serving
  python -m diffserve.launch --model /mnt/models/LLaDA2.0-mini --policy cw-srpt --port 8000

  # Run direct benchmark (no HTTP)
  python -m diffserve.launch --benchmark --sweep --n-reqs 24 --arrival-rate 10

  # Run benchmark with Azure trace
  python -m diffserve.launch --benchmark --trace azure --n-reqs 64
"""

import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diffserve")


def main():
    parser = argparse.ArgumentParser(
        description="DiffServe: Online Serving for Diffusion LLMs")
    subparsers = parser.add_subparsers(dest="command")

    # ─── serve command ────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Start HTTP server")
    serve_parser.add_argument('--model', default='/mnt/models/LLaDA2.0-mini')
    serve_parser.add_argument('--policy', default='cw-srpt',
                              choices=['fcfs', 'srpt', 'cw-srpt', 'bab-srpt', 'appc'])
    serve_parser.add_argument('--max-batch', type=int, default=8)
    serve_parser.add_argument('--threshold', type=float, default=0.9)
    serve_parser.add_argument('--max-seq-length', type=int, default=512)
    serve_parser.add_argument('--host', default='0.0.0.0')
    serve_parser.add_argument('--port', type=int, default=8000)
    serve_parser.add_argument('--device', default='cuda:0')

    # ─── bench command ────────────────────────────────────────────
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument('--model', default='/mnt/models/LLaDA2.0-mini')
    bench_parser.add_argument('--mode', choices=['direct', 'http'], default='direct')
    bench_parser.add_argument('--target', default='http://localhost:8000')
    bench_parser.add_argument('--n-reqs', type=int, default=32)
    bench_parser.add_argument('--gen-length', type=int, default=128)
    bench_parser.add_argument('--threshold', type=float, default=0.9)
    bench_parser.add_argument('--max-batch', type=int, default=8)
    bench_parser.add_argument('--max-seq-length', type=int, default=512)
    bench_parser.add_argument('--policy', default='cw-srpt',
                              choices=['fcfs', 'srpt', 'cw-srpt', 'bab-srpt', 'appc'])
    bench_parser.add_argument('--sweep', action='store_true')
    bench_parser.add_argument('--trace', choices=['poisson', 'azure'], default='poisson')
    bench_parser.add_argument('--trace-path', default=None)
    bench_parser.add_argument('--arrival-rate', type=float, default=5.0)
    bench_parser.add_argument('--dataset', default='humaneval',
                              choices=['humaneval', 'gsm8k'])
    bench_parser.add_argument('--device', default='cuda:0')
    bench_parser.add_argument('--output', default=None)

    # Parse
    args = parser.parse_args()

    if args.command is None:
        # Default: if --benchmark flag or no command, show help
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        _run_serve(args)
    elif args.command == "bench":
        _run_bench(args)


def _run_serve(args):
    """Start the HTTP serving mode."""
    from .config import DiffServeConfig
    from .model_loader import create_model_runner
    from .engine import DiffServeEngine
    from .api_server import run_server

    config = DiffServeConfig(
        model_path=args.model,
        policy=args.policy,
        max_batch_size=args.max_batch,
        threshold=args.threshold,
        max_seq_length=args.max_seq_length,
        host=args.host,
        port=args.port,
        device=args.device,
    )

    logger.info(f"DiffServe starting...")
    logger.info(f"  Model:     {config.model_path}")
    logger.info(f"  Policy:    {config.policy}")
    logger.info(f"  MaxBatch:  {config.max_batch_size}")
    logger.info(f"  Threshold: {config.threshold}")
    logger.info(f"  Device:    {config.device}")
    logger.info(f"  Endpoint:  http://{config.host}:{config.port}")

    model_runner, tokenizer = create_model_runner(config)
    engine = DiffServeEngine(config, model_runner, tokenizer)
    run_server(engine, config)


def _run_bench(args):
    """Run the benchmark."""
    from .bench_online import main as bench_main
    # Rewrite sys.argv for bench_online's argparse
    bench_args = ['bench_online']
    bench_args.extend(['--mode', args.mode])
    bench_args.extend(['--model', args.model])
    bench_args.extend(['--target', args.target])
    bench_args.extend(['--n-reqs', str(args.n_reqs)])
    bench_args.extend(['--gen-length', str(args.gen_length)])
    bench_args.extend(['--threshold', str(args.threshold)])
    bench_args.extend(['--max-batch', str(args.max_batch)])
    bench_args.extend(['--max-seq-length', str(args.max_seq_length)])
    bench_args.extend(['--policy', args.policy])
    bench_args.extend(['--trace', args.trace])
    bench_args.extend(['--arrival-rate', str(args.arrival_rate)])
    bench_args.extend(['--dataset', args.dataset])
    bench_args.extend(['--device', args.device])
    if args.sweep:
        bench_args.append('--sweep')
    if args.trace_path:
        bench_args.extend(['--trace-path', args.trace_path])
    if args.output:
        bench_args.extend(['--output', args.output])

    old_argv = sys.argv
    sys.argv = bench_args
    try:
        bench_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
