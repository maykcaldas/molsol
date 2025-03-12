# Mol Sol

Backend code used in mol.dev

## Quickstart 🔥

```bash
pip install molsol
```

```python
from molsol import KDESol

model = KDESol()

# predict solubility of a molecule
smiles = "CCO"
solubility = model.predict(smiles)
print(solubility)
```


