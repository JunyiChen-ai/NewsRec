import argparse
import logging
import os
import json
import ast
from utility import utils


def _load_config_file(path):
    """Load a YAML/JSON config file into a flat dict.

    Tries PyYAML if available, then JSON, and finally a simple
    line-based parser that supports key: value pairs and JSON-like
    literals for lists and dicts.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Try PyYAML if installed
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        return data or {}
    except Exception:
        pass
    # Try JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: very simple YAML-like parser: key: value per line
    cfg = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if ':' not in s:
            continue
        key, val = s.split(':', 1)
        key = key.strip()
        val = val.strip()
        # Remove inline comments
        if ' #' in val:
            val = val.split(' #', 1)[0].strip()
        # Try to coerce types
        lowered = val.lower()
        if lowered in ('true', 'false'):
            cfg[key] = lowered == 'true'
            continue
        if lowered in ('null', 'none'):
            cfg[key] = None
            continue
        # Literal for list/dict or quoted string
        if (val.startswith('[') and val.endswith(']')) or \
           (val.startswith('{') and val.endswith('}')) or \
           (val.startswith('"') and val.endswith('"')) or \
           (val.startswith("'") and val.endswith("'")):
            try:
                cfg[key] = ast.literal_eval(val)
                continue
            except Exception:
                pass
        # Try numeric
        try:
            if '.' in val:
                cfg[key] = float(val)
            else:
                cfg[key] = int(val)
            continue
        except Exception:
            pass
        # Fallback string
        cfg[key] = val
    return cfg


def parse_args():
    # First, parse only --config to know where to load defaults from
    config_parser = argparse.ArgumentParser(add_help=False)
    default_cfg = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config_parser.add_argument('--config', type=str, default=default_cfg,
                               help='Path to YAML/JSON config file to load defaults from')
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default=
        "/ads-nfs/t-shxiao/cache/data/Mind_large/",
    )

    parser.add_argument("--filename_pat", type=str, default="ProtoBuf_*.tsv")
    parser.add_argument("--model_dir", type=str, default='./saved_models/')
    parser.add_argument("--npratio", type=int, default=1)
    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)
    parser.add_argument("--enable_shuffle", type=utils.str2bool, default=True)
    parser.add_argument("--enable_prefetch", type=utils.str2bool, default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)

    # model training
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--news_attributes",
        type=str,
        nargs='+',
        default=['title', 'abstract'],
        choices=['title', 'abstract', 'body', 'category', 'domain', 'subcategory'])

    parser.add_argument("--num_words_title", type=int, default=32)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=100)

    parser.add_argument("--user_log_length", type=int, default=100)

    parser.add_argument(
        "--word_embedding_dim",
        type=int,
        default=300,
    )
    parser.add_argument("--news_dim", type=int, default=64)
    parser.add_argument("--demo_dim", type=int, default=64)
    parser.add_argument(
        "--news_query_vector_dim",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--user_query_vector_dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--attention_dims",
        type=int,
        default=20,
    )
    parser.add_argument("--user_log_mask", type=utils.str2bool, default=True)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--save_steps", type=int, default=100000)
    parser.add_argument("--max_steps_per_epoch", type=int, default=1000000)

    parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default=None,
        help="choose which ckpt to load and test"
    )
    # share
    parser.add_argument("--title_share_encoder", type=utils.str2bool, default=False)

    # Turing
    parser.add_argument("--pretreained_model", type=str, default='unilm', choices=['unilm', 'others'])
    parser.add_argument("--pretrained_model_path", type=str, default='../tnlr')
    parser.add_argument("--config-name", type=str, default='unilm2-base-uncased-config.json')
    parser.add_argument("--model_name_or_path", type=str, default='unilm2-base-uncased.bin')
    parser.add_argument("--tokenizer_name", type=str, default='unilm2-base-uncased-vocab.txt')

    parser.add_argument("--num_hidden_layers", type=int, default=-1)

    parser.add_argument("--use_pretrain_news_encoder", type=utils.str2bool, default=False)
    parser.add_argument("--freeze_pretrain_news_encoder", type=utils.str2bool, default=False)

    #new parameters for speedyrec
    parser.add_argument("--warmup", type=utils.str2bool, default=False)
    parser.add_argument("--world_size", type=int, default=-1)
    parser.add_argument("--enable_prefetch_stream", type=utils.str2bool, default=True)
    parser.add_argument("--pretrain_lr", type=float, default=1e-4)
    parser.add_argument("--beta_for_cache", type=float, help='the hyper parameter for the growth rate of lookup  probability', default=0.002)
    parser.add_argument("--max_step_in_cache", type=int, help='\gamma', default=20)
    parser.add_argument("--savename", type=str, default='speedy')
    parser.add_argument("--warmup_step", type=int, default=2000)
    parser.add_argument("--schedule_step", type=int, default=30000)
    parser.add_argument("--test_steps", type=int, default=1000000)
    parser.add_argument("--max_hit_ratio", type=float, default=1)

    # early stopping
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Number of epochs with no improvement to wait before stopping. Set <0 to disable.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                        help="Minimum AUC improvement to count as better.")

    # eval / test only
    parser.add_argument("--test_only", type=utils.str2bool, default=False,
                        help="If True, skip training and run dev evaluation once. Uses --load_ckpt_name if provided.")

    # If a config is supplied, load it and set as defaults before final parse
    cfg = _load_config_file(config_args.config)
    if cfg:
        # Normalize keys: replace hyphens with underscores for argparse compatibility
        norm_cfg = {k.replace('-', '_'): v for k, v in cfg.items()}
        parser.set_defaults(**norm_cfg)

    args = parser.parse_args(remaining_argv)
    logging.info(args)
    return args
