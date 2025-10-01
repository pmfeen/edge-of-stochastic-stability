import re


def sanitize_run_name_part(part: str, max_len: int = 128) -> str:
    """Return a W&B-safe name segment: [A-Za-z0-9._-]+ with dashes for others.

    - Collapse invalid character sequences to a single '-'
    - Trim leading/trailing separators
    - Truncate to max_len
    - Ensure non-empty
    """
    if part is None:
        return ""
    part = str(part).strip()
    # Replace any char not alnum, dash, underscore, or dot with '-'
    part = re.sub(r"[^A-Za-z0-9._-]+", "-", part)
    # Collapse multiple dashes
    part = re.sub(r"-+", "-", part)
    # Trim separators
    part = part.strip("-._")
    # Truncate
    if len(part) > max_len:
        part = part[:max_len]
    return part or "run"


def compose_run_name(args) -> str:
    base = f"{args.dataset}_{args.model}_b{args.batch}_lr{args.lr}"
    base = sanitize_run_name_part(base)
    suffix_raw = getattr(args, 'wandb_name', None)
    suffix = sanitize_run_name_part(suffix_raw) if suffix_raw else ""
    if suffix:
        return f"{base}-{suffix}"
    return base

