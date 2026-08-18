# Usikre snit — stage 04 (2026-08-18)

## Status: ingen mærkede endnu — og det er et hul, ikke et resultat

Alle 8 billeder i pilotmaterialet blev snittet succesfuldt og bekræftet
visuelt ord for ord. **Der er ingen implementeret usikkerheds-flagning
endnu** — `find_snitpunkt` returnerer altid et snit, uanset hvor svag
dalen er, og `snit.csv`s `styrke`-kolonne bruges ikke til at afvise noget.

Dette er en reel mangel, ikke en bekræftelse af, at metoden altid virker:

- Alle 8 billeder er fra samme to bind (273098, 273099), samme to måneder
  (maj-juni 1896), formentlig fotograferet i samme session med ensartede
  forhold. Metoden er IKKE afprøvet på billeder med anden belysning,
  anden bogtilstand, eller hvor teksten står tættere på selve snittet.
- Der findes intet eksempel i pilotmaterialet, hvor metoden fejler — så
  der er intet grundlag for at kalibrere en fornuftig `styrke`-tærskel
  endnu. En tærskel sat nu ville være gættet, ikke begrundet.

## Næste skridt, når flere billeder er hentet

1. Kør `find_snitpunkt` på et bredere udsnit (flere bind, årgange).
2. Gennemse kontaktarkene, find de tilfælde hvor snittet rammer forkert.
3. Brug DE tilfælde til at kalibrere en `styrke`-tærskel eller et andet
   signal, der adskiller sikre fra usikre snit.
4. Først derefter giver `usikre.md` mening som en reel liste.
