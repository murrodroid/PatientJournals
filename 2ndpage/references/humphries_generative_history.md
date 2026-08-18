# Mark Humphries: "Generative History" — LLM-HTR-metoder og benchmarks

Web-research 2026-08-18. Humphries (historieprofessor, Wilfrid Laurier) har
2023-2026 systematisk benchmarket frontier-LLM'er mod Transkribus på
håndskrevne 1700-1900-tals dokumenter og bygget værktøjerne Transcription
Pearl / Archive Studio. Peer-reviewed: Humphries et al., "Unlocking the
Archives", *Historical Methods* 58(3), 2025 (preprint arxiv.org/pdf/2411.03340;
kun andenhånds — PDF/SSRN kunne ikke hentes). Alle Substack-poster herunder er
læst førstehånds af research-agenten.

## Benchmark-metodik (direkte genbrugelig)

- Egen korpus (~50 dokumenter / ~10.000 ord, 1700-1800-tals engelsk),
  kontamineringstjekket. Seneste kørsler: samme sæt 10 gange pr. model
  (500 kørsler) for stabile gennemsnit.
- **To metrikker side om side**: "strict" (alle forskelle tæller) og
  "modified" (ignorerer versaler, tegnsætning og historisk stavevariation) —
  modified er hans foretrukne, da streng CER overdriver reelle fejl på
  1800-tals tekst.
- Menneskelige baselines som loft: professionelle ~0,2-1,9 % CER,
  ikke-professionelle ~4-10 % WER.

## Modelresultater (modified CER, hans testsæt — flygtige!)

| Dato | Setup | Mod. CER |
|---|---|---|
| Nov 2024 | GPT-4o / Claude Sonnet 3.5 direkte | 5,7-7,0 % |
| Nov 2024 | Transkribus → Sonnet 3.5-korrektion | **1,8 %** (bedst dengang) |
| Nov 2025 | **Gemini 3 Pro** | **0,69 %** (strict 1,67 %) |
| Nov 2025 | Claude Opus 4.5 | 2,53 % |
| Nov 2025 | OpenAI-model (unavngiven) | 11,9 % |
| Nov 2025 | Transkribus, finetunet på materialet | ~3 % |

Spring på 50-70 % fejlreduktion pr. modelgeneration på uger ⇒ **eval-rammen
skal designes til genkørsel, ikke til at kåre én vinder én gang.**

## Teknikker der virkede

- **Transcribe-then-correct**: førstepasning (anden LLM eller Transkribus)
  gives som kontekst til en LLM der RETTER frem for at transskribere forfra —
  hans bedste 2024-opskrift.
- **Batch/parallel-kald**: ~$0,014/side vs. ~$0,27/side for
  Transkribus-baseret arbejdsgang (~50x billigere).
- **Simple prompts i almindeligt sprog** — avanceret prompt-engineering var
  ikke nødvendigt.
- **Inference-indstillinger (Gemini 3)**: temperatur 0, HØJ billedopløsning,
  minimal thinking — opløsning betød mere end ekstra ræsonnement.
- To-modellers triage i stor skala (billig model sorterer, dyr model
  ekstraherer) som omkostningsteknik.

## Uoverensstemmelses-flagning (kernen for os)

["When Models Disagree… Transcription Accuracy Improves Significantly"](https://generativehistory.substack.com/p/when-models-disagree-transcription)
(27. maj 2026): basis-transskription med Gemini 3 Pro; to ANDRE modelfamilier
(Gemini 3.5 Flash + Claude Opus 4.7) gen-transskriberer; kun ord/passager hvor
modellerne er uenige (4 % af teksten) sendes til menneskelig review.
Resultat: modified WER 1,33 % → 0,33 %, CER 0,69 % → 0,23 %; **76 % af
restfejlene fanget** med menneskelig opmærksomhed på kun ~4 % af teksten.
Pointe: den anden model skal ikke være bedre — bare fra en anden modelfamilie,
så fejlmønstrene ikke korrelerer. (= rollen lead tiltænkte Claude.)
Restfejl efter korrektion: stavemodernisering, beskadiget tekst, tal/datoer.

## Transkribus vs. frontier-LLM'er

Generisk Transkribus: 8-25 % CER. Finetunet: ~3 % — stadig slået af zero-shot
Gemini 3 (0,69 %). Hans konklusion (via Suttons "Bitter Lesson"): generalist-
modeller har overhalet specialbygget HTR; Transkribus' bedste rolle blev
billig førstepasning til LLM-korrektion.

## Nøgleposter

- [Introducing Transcription Pearl](https://generativehistory.substack.com/p/introducing-transcription-pearl) (1. nov 2024)
- [Has Google Quietly Solved Two of AI's Oldest Problems?](https://generativehistory.substack.com/p/has-google-quietly-solved-two-of) (17. okt 2025)
- [Gemini 3 Solves Handwriting Recognition and it's a Bitter Lesson](https://generativehistory.substack.com/p/gemini-3-solves-handwriting-recognition) (25. nov 2025)
- [When Models Disagree…](https://generativehistory.substack.com/p/when-models-disagree-transcription) (27. maj 2026)
- [Making the Infeasible Practical in Historical Research](https://generativehistory.substack.com/p/making-the-infeasible-practical-in) (31. maj 2024)
