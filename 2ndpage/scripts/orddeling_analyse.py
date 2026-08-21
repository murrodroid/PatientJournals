"""Hvor ofte deler transskribenten et ord over to linjer UDEN bindestreg?

Vi kan ikke slaa op i en ordbog, men vi kan bruge korpusset som ordbog:
findes stumpen "Inspira" aldrig som selvstaendigt ord i de 168 sider, mens
"Inspiration" gør, saa er linjeskiftet efter al sandsynlighed midt i et ord.
Det er et skoen, ikke en facitliste -- men det giver en stoerrelsesorden.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ORD = re.compile(r"[A-Za-zÆØÅæøåÉéÜü]+")
sider = [json.loads(l) for l in open("stages/02_facit/output/facit.jsonl", encoding="utf-8")]

# Ordforrådet i hele materialet, med små bogstaver.
ordbog: Counter[str] = Counter()
for s in sider:
    for o in ORD.findall(s["alt_fladet"]):
        ordbog[o.lower()] += 1

med_bindestreg = 0
uden_bindestreg = []
almindeligt_linjeskift = 0

for s in sider:
    linjer = [l.strip() for l in s["alt_linjer"]]
    for i in range(len(linjer) - 1):
        a, b = linjer[i], linjer[i + 1]
        if not a or not b:
            continue
        if a.endswith("-"):
            med_bindestreg += 1
            continue
        sidste = ORD.findall(a)
        foerste = ORD.findall(b)
        if not sidste or not foerste:
            continue
        hale, hoved = sidste[-1], foerste[0]
        # Kun relevant naar naeste linje starter med lille bogstav og linjen
        # ikke slutter paa tegnsaetning.
        if not a[-1].isalpha() or not hoved[0].islower():
            almindeligt_linjeskift += 1
            continue
        samlet = (hale + hoved).lower()
        # Stumpen findes aldrig alene, men det samlede ord findes: staerkt tegn.
        if ordbog[hale.lower()] <= 1 and ordbog[samlet] >= 1:
            uden_bindestreg.append((s["image_name"], hale, hoved, ordbog[samlet]))
        else:
            almindeligt_linjeskift += 1

print(f"linjepar i alt undersoegt:        {med_bindestreg + almindeligt_linjeskift + len(uden_bindestreg)}")
print(f"linjer der slutter med bindestreg (markeret orddeling): {med_bindestreg}")
print(f"formodet UMARKERET orddeling:    {len(uden_bindestreg)}")
print(f"almindelige linjeskift:          {almindeligt_linjeskift}")
print()
print("Eksempler paa formodet umarkeret orddeling:")
for navn, hale, hoved, n in uden_bindestreg[:20]:
    print(f"  {navn}  {hale!r} + {hoved!r} -> {(hale + hoved)!r} (findes {n} gange i korpus)")
print()
tegn = sum(len(s["alt_fladet"]) for s in sider)
print(f"tegn i alt i facit: {tegn}")
print(f"ét ekstra mellemrum pr. tilfaelde svarer til {len(uden_bindestreg) / tegn * 100:.3f} % af tegnene")
