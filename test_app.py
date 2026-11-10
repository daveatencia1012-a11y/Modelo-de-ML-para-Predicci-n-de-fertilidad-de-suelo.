"""Verificaciones de contrato, datos reales y flujo Streamlit: python -m unittest -v."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch
import json
import hashlib

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / 'app.py').read_text()
DATA = (ROOT / 'dataset1.csv').read_bytes()
# Importa las funciones reales sin ejecutar la interfaz al descubrir pruebas.
tree = ast.parse(SOURCE)
tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                                                  ('BASE_DIR', 'COMPONENTES', 'CLASIFICACIONES') for t in n.targets)]
API = {'__file__': str(ROOT / 'app.py')}
exec(compile(tree, str(ROOT / 'app.py'), 'exec'), API)


def app_with(data):
    source = SOURCE.replace('datos = cargar_datos_bytes()',
                            f'cargar_datos_bytes()\ndatos = {data!r}')
    return AppTest.from_string(source, default_timeout=30).run()


class FertilidadTests(unittest.TestCase):
    def test_real_dataset_and_prediction_contract(self):
        model = API['entrenar'](DATA)
        self.assertEqual(len(model['df']), 880)
        self.assertEqual(len(model['cols']), 12)
        sample = model['X'].quantile(.25).to_frame().T.astype(model['X'].dtypes)
        self.assertEqual(list(sample), list(model['scaler'].feature_names_in_))
        self.assertTrue(sample.dtypes.equals(model['X'].dtypes))
        proba = model['clf'].predict_proba(model['scaler'].transform(sample))[0]
        self.assertAlmostEqual(proba.sum(), 1)
        self.assertEqual(list(model['clf'].classes_), [0, 1, 2])

    def test_invalid_datasets(self):
        df = pd.read_csv(ROOT / 'dataset1.csv')
        variants = [df.drop(columns='N'), df.assign(extra=1), df.assign(Output=4),
                    df.assign(N=np.nan), df.assign(P=np.inf), df.assign(K='texto'),
                    df.iloc[:0], df.iloc[:2]]
        for invalid in variants:
            with self.subTest(columns=list(invalid), rows=len(invalid)):
                with self.assertRaises(ValueError):
                    API['validar_dataset'](invalid.to_csv(index=False).encode())
        with self.assertRaises(ValueError):
            API['validar_dataset'](DATA.replace(b'N,P,', b'N,N,', 1))

    def test_ui_prediction_and_invalidation(self):
        at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.selectbox), 12)
        df = pd.read_csv(ROOT / 'dataset1.csv')
        for widget, c in zip(at.selectbox, API['COMPONENTES']):
            values = df[c].quantile([.25, .5, .75]).tolist()
            self.assertEqual(widget.options, [f"{label} ({API['numero_visible'](v)})" for label, v in
                                             zip(['Baja', 'Media', 'Alta'], values)])
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertIn('ultima_pred', at.session_state)
        self.assertEqual(len(at.get('plotly_chart')), 7)
        at.selectbox[0].select(0).run()
        self.assertNotIn('ultima_pred', at.session_state)
        self.assertFalse(at.exception)

    def test_missing_invalid_and_replacement(self):
        for data in (None, b'N,Output\n1,0'):
            at = app_with(data)
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            self.assertEqual(len(at.selectbox), 0)
            self.assertNotIn('ultima_pred', at.session_state)
        changed = pd.read_csv(ROOT / 'dataset1.csv').assign(N=lambda df: df.N + 100)
        data = changed.to_csv(index=False).encode()
        at = app_with(data)
        self.assertFalse(at.exception)
        self.assertIn(f"({API['numero_visible'](changed.N.median())})", at.selectbox[0].options[1])
        original = API['entrenar'](DATA)
        updated = API['entrenar'](data)
        self.assertIsNot(original['clf'], updated['clf'])
        self.assertAlmostEqual(updated['X'].N.median(), original['X'].N.median() + 100)

    def test_dataset_change_in_same_session(self):
        source = SOURCE.replace('datos = cargar_datos_bytes()',
                                f'cargar_datos_bytes()\ndatos = st.session_state.get("test_csv", {DATA!r})')
        at = AppTest.from_string(source, default_timeout=30).run()
        at.button[0].click().run()
        self.assertIn('ultima_pred', at.session_state)
        old_key = at.selectbox[0].key
        changed = pd.read_csv(ROOT / 'dataset1.csv').assign(N=lambda df: df.N + 100)
        at.session_state['test_csv'] = changed.to_csv(index=False).encode()
        at.run()
        self.assertFalse(at.exception)
        self.assertNotIn('ultima_pred', at.session_state)
        self.assertNotEqual(at.selectbox[0].key, old_key)
        self.assertEqual(at.selectbox[0].value, 1)
        at.button[0].click().run()
        at.session_state['test_csv'] = b'invalid'
        at.run()
        self.assertNotIn('ultima_pred', at.session_state)
        self.assertEqual(len(at.get('plotly_chart')), 0)

    def test_exact_display_and_minority_metrics(self):
        for value in (1.2345678901234567, 0.30000000000000004, 1e-15):
            self.assertEqual(float(API['numero_exacto'](value)), value)
        model = API['entrenar'](DATA)
        self.assertEqual(model['reporte']['2']['support'], 8)
        self.assertEqual(sum(model['reporte'][str(c)]['support'] for c in (0, 1, 2)), 176)

    def test_metadata_bound_to_dataset_and_probability_order(self):
        digest = hashlib.sha256(DATA).hexdigest()
        # Fixture de validación técnica, no evidencia científica del CSV real.
        config = {'dataset_sha256': digest, 'fuente': 'fuente de prueba',
                  'evidencia': 'fixture', 'correspondencia':
                  {'0': 'menos fértil', '1': 'fértil', '2': 'altamente fértil'}}
        with patch.object(Path, 'exists', return_value=True), patch.object(Path, 'read_text', return_value=json.dumps(config)):
            self.assertEqual(API['cargar_clases']('otro-csv'), {})
            verified = API['cargar_clases'](digest)
            self.assertEqual(len(verified), 3)
        with patch.dict(API, {'CLASES': verified}):
            fig = API['fig_probabilidades']([2, 0, 1], [.6, .1, .3])
            self.assertEqual([t.marker.color for t in fig.data], ['#16A34A', '#DC3545', '#EAB308'])
            self.assertEqual([float(t.y[0]) for t in fig.data], [.6, .1, .3])
            self.assertIn('ALTAMENTE', fig.data[0].name)
        with patch.dict(API, {'CLASES': {}}):
            fig = API['fig_probabilidades']([2, 0, 1], [.6, .1, .3])
            self.assertTrue(all(t.marker.color == '#64748b' for t in fig.data))

    def test_prediction_error_is_recoverable(self):
        source = SOURCE.replace("muestra_scaled = model['scaler'].transform(muestra)",
                                "raise ValueError('fallo de prueba')")
        at = AppTest.from_string(source.replace('datos = cargar_datos_bytes()',
                                                f'cargar_datos_bytes()\ndatos = {DATA!r}'), default_timeout=30).run()
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertNotIn('ultima_pred', at.session_state)
        self.assertEqual(len(at.selectbox), 12)


if __name__ == '__main__':
    unittest.main()
