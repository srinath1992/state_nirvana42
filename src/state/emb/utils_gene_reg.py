import torch
from typing import Dict, Tuple, Any

from .utils import get_embedding_cfg


def _infer_index_mapping(obj: Any) -> Dict[str, int]:
    """
    Try to infer a mapping from gene_name -> index from a torch-loaded object.
    Supports:
    - dict[str -> int]
    - dict[int -> str]
    - list[str] (index = position)
    - tuple/list of (names, indices) in either order
    """
    if isinstance(obj, dict):
        # decide key/value orientation
        k0 = next(iter(obj.keys()))
        if isinstance(k0, str) and isinstance(next(iter(obj.values())), int):
            return obj  # name -> index
        if isinstance(k0, int) and isinstance(next(iter(obj.values())), str):
            return {v: k for k, v in obj.items()}  # invert to name -> index
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return {}
        # list[str]
        if all(isinstance(x, str) for x in obj):
            return {name: i for i, name in enumerate(obj)}
        # tuple/list of (names, indices)
        if len(obj) == 2:
            a, b = obj
            if isinstance(a, (list, tuple)) and all(isinstance(x, str) for x in a) and isinstance(b, (list, tuple)):
                return {name: int(i) for name, i in zip(a, b)}
            if isinstance(b, (list, tuple)) and all(isinstance(x, str) for x in b) and isinstance(a, (list, tuple)):
                return {name: int(i) for name, i in zip(b, a)}
    raise ValueError("Unsupported ds_emb_mapping format; expected dict, list[str], or (names, indices).")


def build_gene_reg_table(cfg) -> torch.Tensor:
    """
    Build a [num_tokens, gene_reg_dim] tensor aligned to the same token index
    space as the protein embedding table.

    cfg.se.gene_reg_feature_file: path to a torch file with dict: name -> FloatTensor[dim]
    embeddings.current.ds_emb_mapping: path to mapping defining index order
    """
    emb_cfg = get_embedding_cfg(cfg)
    num_tokens = int(emb_cfg["num"])
    reg_dim = int(cfg.se.gene_reg_dim)

    # load features
    reg_dict: Dict[str, torch.Tensor] = torch.load(cfg.se.gene_reg_feature_file, map_location="cpu")
    # load mapping
    ds_map_obj = torch.load(emb_cfg["ds_emb_mapping"], map_location="cpu")
    name_to_index = _infer_index_mapping(ds_map_obj)

    table = torch.zeros((num_tokens, reg_dim), dtype=torch.float32)
    hits = 0
    for gene_name, vec in reg_dict.items():
        idx = name_to_index.get(gene_name)
        if idx is None:
            continue
        try:
            table[int(idx)] = vec.detach().to(dtype=torch.float32, device="cpu")
            hits += 1
        except Exception:
            continue
    if hits == 0:
        raise ValueError("No gene_reg features matched ds_emb_mapping indices. Check alignment and names.")
    return table


