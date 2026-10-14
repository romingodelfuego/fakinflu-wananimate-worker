#!/usr/bin/env python3
"""Garde-fou de BUILD, 100% statique (aucun import de torch/comfy, donc aucune init CUDA).

Analyse un noeud ComfyUI (AST), releve chaque acces `comfy.<module>.<attr>`, et verifie que
`<attr>` est bien defini dans la source comfy correspondante de l'image. Attrape une API
inventee (ex. `comfy.model_management.intermediate_dtype()` qui n'existe pas) AVANT le deploiement,
sans jamais toucher au GPU. Auto-maintenu : rien a lister a la main.

Usage : check_comfy_api.py <node.py> <dossier_comfy>
"""
import ast
import pathlib
import re
import sys


def defined(attr, text):
    """Vrai si `attr` est defini (def/class), assigne au niveau module, ou importe dans `text`."""
    pats = [rf"\bdef\s+{attr}\b", rf"\bclass\s+{attr}\b", rf"(?m)^\s*{attr}\s*=",
            rf"\bas\s+{attr}\b", rf"\bimport\b[^\n]*\b{attr}\b"]
    return any(re.search(p, text) for p in pats)


def main(node_file, comfy_dir):
    src = pathlib.Path(node_file).read_text()
    tree = ast.parse(src)  # une SyntaxError fait echouer le build ici
    used = set()
    for n in ast.walk(tree):
        # motif comfy.<module>.<attr>
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute)
                and isinstance(n.value.value, ast.Name) and n.value.value.id == "comfy"):
            used.add((n.value.attr, n.attr))
    missing = []
    for mod, attr in sorted(used):
        p = pathlib.Path(comfy_dir, mod + ".py")
        if not p.exists():
            missing.append(f"comfy.{mod} (fichier absent: {p})")
        elif not defined(attr, p.read_text()):
            missing.append(f"comfy.{mod}.{attr}")
    if missing:
        print("ECHEC garde-fou API comfy — introuvable:", ", ".join(missing))
        return 1
    print("garde-fou API comfy OK:", sorted(f"comfy.{m}.{a}" for m, a in used))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
