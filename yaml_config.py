"""Strict 3-tier argparse + YAML config loader.

Precedence:  argparse defaults  <  --config yaml  <  CLI flags

Unknown YAML keys raise SystemExit — typos like `batchsize: 16` fail loudly
rather than silently leaving the default in place. Used by train.py and the
tools/ entry points so the same `configs/default.yaml` drives all of them.
"""
import argparse
import os
import sys


def parse_args_with_yaml(build_parser, argv=None):
    """Parse argv with full YAML support.

    `build_parser` must be a 0-arg callable returning a fresh ArgumentParser
    (we call it twice: once to discover dests + once to re-parse with the
    yaml-merged defaults).
    """
    if argv is None:
        argv = sys.argv[1:]

    ap0 = build_parser()
    defaults = vars(ap0.parse_args([]))   # tier 1: every dest with its argparse default

    pre, _ = ap0.parse_known_args(argv)
    yaml_cfg = {}
    if getattr(pre, 'config', None):
        try:
            import yaml
        except ImportError as e:
            raise SystemExit('YAML config requested but PyYAML is not installed: {}'.format(e))
        if not os.path.isfile(pre.config):
            raise SystemExit('--config file not found: {}'.format(pre.config))
        with open(pre.config) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        if not isinstance(yaml_cfg, dict):
            raise SystemExit('--config must be a YAML mapping, got: {}'.format(type(yaml_cfg).__name__))
        unknown = set(yaml_cfg) - set(defaults)
        if unknown:
            raise SystemExit('Unknown keys in {}: {} (allowed dests: {})'.format(
                pre.config, sorted(unknown), sorted(defaults)))
        defaults.update(yaml_cfg)

    # tier 3: CLI — re-parse with yaml-merged defaults so any flag the user typed wins.
    ap = build_parser()
    ap.set_defaults(**defaults)
    args = ap.parse_args(argv)
    args._yaml_keys = sorted(yaml_cfg)
    return args


def dump_resolved(args, out_path, exclude=('config',)):
    """Dump the resolved args namespace to a YAML file so the run is reproducible
    from artifacts. Skips internal fields and unserializable values."""
    try:
        import yaml
    except ImportError:
        return   # silent — caller decides whether to require yaml

    def _safe(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _safe(x) for k, x in v.items()}
        return repr(v)

    payload = {}
    for k, v in sorted(vars(args).items()):
        if k.startswith('_') or k in exclude:
            continue
        payload[k] = _safe(v)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.safe_dump(payload, f, sort_keys=False)
