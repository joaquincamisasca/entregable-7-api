"""Tests del cálculo de percentiles del script de prueba de carga."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("load_test", Path(__file__).parent.parent / "scripts" / "load_test.py")
load_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(load_test)


def test_p95_con_cinco_muestras_es_la_mas_lenta():
    assert load_test.percentil([3.0, 1.0, 2.0, 5.0, 4.0], 95) == 5.0


def test_p50_es_la_mediana():
    assert load_test.percentil([3.0, 1.0, 2.0, 5.0, 4.0], 50) == 3.0


def test_una_sola_muestra():
    assert load_test.percentil([2.5], 95) == 2.5
