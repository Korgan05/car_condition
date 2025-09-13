from typing import Dict


def format_metrics(metrics: Dict) -> str:
    def fmt(d):
        return {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in d.items()}
    out = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            out[k] = fmt(v)
        else:
            out[k] = v
    return str(out)
