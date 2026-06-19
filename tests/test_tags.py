"""Tests unitaires déterministes du flag `compress(tags=True)` et de `_contract_tags`.

Aucune dépendance à des repos clonés : tout est du code inline.
Couvre le contrat des tags raises/yields/warns + la non-régression de S5 pur.
(Le test d'overfit sur 6 vrais repos vit dans `test_tags_overfit.py`.)
"""
import json
import unittest

import ast_extractor as ae


def _needs(bundle, fn):
    """Récupère le dict du `# needs` de la fonction `fn` dans un bundle, ou {}."""
    lignes = bundle.splitlines()
    for i, l in enumerate(lignes):
        if l.lstrip().startswith(f"def {fn}(") or l.lstrip().startswith(f"async def {fn}("):
            for suite in lignes[i + 1:i + 4]:
                j = suite.find("# needs: ")
                if j != -1:
                    return json.loads(suite[j + len("# needs: "):])
            return {}
    raise AssertionError(f"fonction {fn} absente du bundle")


# Une DATA module-level rend les fonctions qui la lisent -> TOOL (donc strippées).
DATA_SRC = "CFG = {'p': 1}\n"


class TestContractTags(unittest.TestCase):

    def test_default_is_pure_s5_no_contract_keys(self):
        """tags=False (défaut) : jamais de clé raises/yields/warns."""
        src = DATA_SRC + (
            "def f(x):\n"
            "    if x: raise ValueError('bad')\n"
            "    return CFG[x]\n"
        )
        besoin = _needs(ae.compress(src), "f")          # défaut
        self.assertEqual(set(besoin) & {"raises", "yields", "warns"}, set())
        self.assertIn("data", besoin)                   # S5 pur intact

    def test_raises_captured_with_types_and_deduped(self):
        src = DATA_SRC + (
            "def f(x):\n"
            "    if x: raise ValueError('a')\n"
            "    if not x: raise KeyError('b')\n"
            "    if x == 2: raise ValueError('again')\n"
            "    return CFG[x]\n"
        )
        besoin = _needs(ae.compress(src, tags=True), "f")
        self.assertEqual(besoin.get("raises"), ["ValueError", "KeyError"])  # ordre + dédup

    def test_yields_marks_stripped_generator(self):
        src = DATA_SRC + (
            "def g(xs):\n"
            "    for x in xs:\n"
            "        yield CFG[x]\n"          # lit CFG -> TOOL -> strippée
        )
        besoin = _needs(ae.compress(src, tags=True), "g")
        self.assertTrue(besoin.get("yields"))

    def test_warns_captured(self):
        src = DATA_SRC + (
            "import warnings\n"
            "def f(x):\n"
            "    warnings.warn('deprecated')\n"
            "    return CFG[x]\n"
        )
        besoin = _needs(ae.compress(src, tags=True), "f")
        self.assertIn("warnings.warn", besoin.get("warns", []))

    def test_nested_function_contracts_not_attributed_to_parent(self):
        """Un raise/yield d'une closure ne doit PAS remonter au parent strippé."""
        src = DATA_SRC + (
            "def outer(x):\n"
            "    def helper():\n"
            "        raise KeyError('nested')\n"
            "        yield 1\n"
            "    return CFG[x]\n"             # outer lit CFG -> TOOL -> strippée
        )
        besoin = _needs(ae.compress(src, tags=True), "outer")
        self.assertNotIn("raises", besoin)
        self.assertNotIn("yields", besoin)

    def test_pure_function_keeps_body_regardless_of_tags(self):
        """Fonction PURE : corps gardé, donc pas de ligne # needs (rien à tagger)."""
        src = "def add(a, b):\n    return a + b\n"      # aucune dep -> PURE
        bundle = ae.compress(src, tags=True)
        self.assertIn("return a + b", bundle)            # corps présent
        self.assertNotIn("# needs:", bundle)

    # Le coût en tokens des tags (marginal : −0,5 pt agrégé) se mesure à l'échelle
    # d'un vrai repo, pas en unitaire : voir test_tags_overfit.py.


CLASS_SRC = (
    "class Cache:\n"
    "    LIMIT = 100\n"
    "    def __init__(self, n):\n"
    "        self.n = n\n"
    "        self._data = {}\n"
    "    def get(self, k):\n"
    "        if k not in self._data: raise KeyError(k)\n"
    "        return self._data[k]\n"
    "    def put(self, k, v):\n"
    "        self._data[k] = v\n"
    "        if len(self._data) > self.n: self._evict()\n"
    "    def _evict(self):\n"
    "        self._data.popitem()\n"
)


class TestResolveSelf(unittest.TestCase):

    def test_default_ignores_self_no_regression(self):
        """resolve_self=False (défaut) : aucune dep self.* dans les # needs."""
        bundle = ae.compress(CLASS_SRC, tags=True)            # défaut
        self.assertNotIn("self.", bundle.split("# needs")[-1] if "# needs" in bundle else "")
        # méthodes OO gardées entières (PURE-par-accident) en S5 pur
        self.assertIn("return self._data[k]", bundle)

    def test_self_methods_and_attrs_resolved(self):
        bundle = ae.compress(CLASS_SRC, tags=True, resolve_self=True)
        needs = _needs(bundle, "put")
        self.assertIn("self._data", needs.get("data", []))    # attribut
        self.assertIn("self._evict", needs.get("interne", []))  # méthode soeur -> ORCH-like

    def test_self_signal_is_anchored_in_bundle(self):
        """Toute dep self.x nommée est ancrée : __init__ gardé + def présent."""
        bundle = ae.compress(CLASS_SRC, tags=True, resolve_self=True)
        self.assertIn("self._data = {}", bundle)              # __init__ verbatim
        self.assertIn("self.n = n", bundle)
        self.assertIn("def _evict(self):", bundle)            # méthode soeur présente
        self.assertIn("LIMIT = 100", bundle)                  # attribut de classe gardé

    def test_init_body_kept_verbatim(self):
        bundle = ae.compress(CLASS_SRC, resolve_self=True)
        self.assertNotIn("def __init__(self, n):\n        # needs", bundle)


# cible TOOL (utilise une DATA) appelant un helper TOOL -> sans full_targets,
# la cible ELLE-MÊME serait strippée ; on vérifie qu'elle reste verbatim.
EXPAND_SRC = (
    "THRESHOLD = 42\n"
    "GAIN = 10\n"
    "def helper(n):\n"
    "    \"\"\"Apply gain.\"\"\"\n"
    "    return n * GAIN\n"
    "def target(xs):\n"
    "    \"\"\"Sum gained values above threshold.\"\"\"\n"
    "    total = 0\n"
    "    for x in xs:\n"
    "        if x > THRESHOLD:\n"
    "            total += helper(x)\n"
    "    return total\n"
)


class TestExpandLossless(unittest.TestCase):

    def test_target_kept_verbatim(self):
        from combine import expand
        ctx = expand(EXPAND_SRC, {"target"})              # full_targets=True défaut
        self.assertIn("for x in xs:", ctx)                # corps complet présent
        self.assertIn("total += helper(x)", ctx)
        self.assertIn("return total", ctx)

    def test_closure_dependency_is_compressed(self):
        from combine import expand
        ctx = expand(EXPAND_SRC, {"target"})
        self.assertIn("def helper(n):", ctx)              # décor : signature présente
        self.assertNotIn("return n * GAIN", ctx)          # mais corps strippé
        self.assertIn("# needs", ctx)

    def test_vocabulary_anchored(self):
        from combine import expand
        ctx = expand(EXPAND_SRC, {"target"})
        self.assertIn("THRESHOLD = 42", ctx)              # dep de la cible
        self.assertIn("GAIN = 10", ctx)                   # dep de la clôture

    def test_target_roundtrips_exactly(self):
        """La cible verbatim se reparse à l'identique (lossless réel)."""
        import ast as _ast
        from combine import expand
        ctx = expand(EXPAND_SRC, {"target"})
        orig = next(n for n in _ast.parse(EXPAND_SRC).body
                    if isinstance(n, _ast.FunctionDef) and n.name == "target")
        got = next(n for n in _ast.parse(ctx).body
                   if isinstance(n, _ast.FunctionDef) and n.name == "target")
        self.assertEqual(_ast.unparse(got), _ast.unparse(orig))

    def test_full_targets_false_strips_target(self):
        from combine import expand
        ctx = expand(EXPAND_SRC, {"target"}, full_targets=False)
        self.assertNotIn("total += helper(x)", ctx)       # cible strippée (ancien mode)


class TestTokenCounter(unittest.TestCase):

    def test_cost_formula_matches_haiku_rates(self):
        """$1/MTok input + $5/MTok output sur une réponse type de 500 tokens."""
        import token_counter as tc
        s = tc.compute_stats(1_000_000, 0)
        self.assertAlmostEqual(s["cost_before_usd"], 1.0 + 500 * 5e-6, places=9)
        self.assertAlmostEqual(s["cost_after_usd"], 0.0 + 500 * 5e-6, places=9)

    def test_reduction_and_savings(self):
        import token_counter as tc
        s = tc.compute_stats(1000, 400)
        self.assertAlmostEqual(s["reduction_pct"], 60.0, places=6)
        # économie = (1000-400) tokens d'input * $1e-6
        self.assertAlmostEqual(s["savings_usd"], 600 * 1e-6, places=9)

    def test_zero_before_no_crash(self):
        import token_counter as tc
        self.assertEqual(tc.compute_stats(0, 0)["reduction_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
