# BayesWHAM Python 3 compatibility port

`BayesWHAM.py` is a compatibility copy of Andrew L. Ferguson's original
BayesWHAM v1.0 Python 2.7 implementation. The untouched upstream repository is
stored at `../bayeswham_python`.

- Upstream repository: https://bitbucket.org/andrewlferguson/bayeswham_python
- Upstream commit: `a694636ce1ecbdc052dd143195734487dd1ac445`
- Upstream license: MIT (`../bayeswham_python/LICENSE.md`)
- Initial compatibility conversion: Python's standard `lib2to3` tool

No scientific or statistical changes have been made. In addition to the syntax
conversion, the temporary bin-index array was changed from unsigned to signed
integer storage because NumPy 2 rejects adding the neighbor offset `-1` to an
unsigned value. This preserves the original indexing intent.
The saved-sample index also uses explicit integer division to preserve Python
2's behavior under Python 3.
