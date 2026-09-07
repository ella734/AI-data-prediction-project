"""
src/utils.py
─────────────────────────────────────────────────────────────────────────────
Utilitaires communs : configuration du logging, reproductibilité, timing.
"""

import os
import sys
import time
import logging
import random
import functools
from datetime import datetime
from typing import Callable, Any

import numpy as np


def setup_logging(level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Configure le logging global du projet.

    Args:
        level: Niveau de log (ex: logging.DEBUG, logging.INFO).
        log_file: Chemin optionnel vers un fichier de log.

    Returns:
        Logger racine configuré.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger()


def set_seed(seed: int = 42) -> None:
    """
    Fixe toutes les graines aléatoires pour la reproductibilité.

    Args:
        seed: Valeur de la graine.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (si disponible)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def timer(func: Callable) -> Callable:
    """Décorateur qui mesure et affiche le temps d'exécution d'une fonction."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.getLogger(__name__).info(
            f"[TIMER] {func.__qualname__}() terminé en {elapsed:.3f}s"
        )
        return result
    return wrapper


def get_run_id() -> str:
    """Retourne un identifiant unique de run basé sur l'horodatage."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_header(title: str, width: int = 60) -> None:
    """Affiche un titre formaté dans la console."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def check_gpu_availability() -> str:
    """Vérifie la disponibilité du GPU et retourne une description."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"GPU disponible : {gpu_name} ({mem:.1f} GB VRAM)"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon MPS disponible"
        else:
            return "Aucun GPU détecté – utilisation du CPU"
    except ImportError:
        return "PyTorch non installé"
