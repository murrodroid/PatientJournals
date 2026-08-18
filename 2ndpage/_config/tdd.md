# Testregler for 2ndpage

## Hvad der SKAL testes

Deterministisk logik — kode hvor samme input altid giver samme output:

- **Facit-læseren**: RTF-afkodning, opdeling i `[page]`-blokke, klamme-
  konventioner (`[?]`, `[crossed out]…[written instead]…`,
  understregningsnoter, positionsmærker), orddeling hen over linjeskift.
- **Måleapparatet**: normalisering, tegnfejl, ordfejl, justering,
  hallucinationstjek.
- **Billedforberedelsen**: bogryg-snit, frasortering af naboblade,
  beskæring. Koordinater er tal — de kan testes.

## Hvad der IKKE skal testes med faste tests

Modelkald og modelsvar. De varierer mellem kørsler og mellem modelversioner.
De dokumenteres i stedet som **resultater** i stagens `output/`, med angivelse
af model, promptversion, dato og indstillinger, så en kørsel kan genfindes.

## Testniveauer

| Niveau | Hvad | Krav |
|---|---|---|
| Enhed | Én funktion, konstrueret input | Skal dække kant- og fejltilfælde, ikke kun det normale forløb |
| Kontrakt | Et modul mod en aftalt fil- eller dataform | Skal fejle, hvis formen ændrer sig |
| Struktur | ICM-strukturen selv | `tests/test_icm_structure.py` |

## Regler

- **Skriv aldrig en triviel test.** Et rent gennemløbstjek uden logik, eller
  en test der kun bekræfter hvad Python selv garanterer, er værdiløs.
- **En leveret test skal være set fejle.** Genindfør fejlen, se testen blive
  rød, gendan koden. Ellers ved vi ikke, at den vogter noget.
- **Testdata må gerne være konstrueret.** Data der fremkalder en bestemt fejl
  er sjældent repræsentative — det er meningen.
- **Determinisme**: undgå at iterere over mængder og ordbøger, hvor
  rækkefølgen påvirker output. Det har givet ikke-reproducerbare resultater i
  tidligere projekter. Sortér udtrykkeligt.

## Definition of Done for en stage

1. Alle filer nævnt i stagens `Outputs`-tabel findes i `output/`.
2. Stagens `Test Contract` er opfyldt, og testene kører grønt.
3. Outputtet er gennemset af et menneske.
4. `PROGRESS.md` er opdateret, og en beslutning af betydning er skrevet ind i
   rod-`CONTEXT.md`.
