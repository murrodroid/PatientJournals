Denne kontekstfil er et oplæg til AI agenter til at beskrive dette projekts første mål

Dette projekt har til formål at udvikle en løsning, der kan transskribere håndskrevne sider i patientjournaler fra blegdamshospitalet
Journalernes forsider er transskriberet andetsteds. Andensiderne indeholder mange modtagenotater om patienterne, som er interessante for vores forskningsprojekt. Resten af siderne kan være mere komplekse med mindre brugbar info - ikke relevant pt.
Forsiderne er transskriberede ved generelle prompts om enkelte objekter på siderne via https://github.com/murrodroid/PatientJournals
Jeg fornemmer ikke at vi kan bruge samme opdelte og kontekstafhængige prompts til andensiderne, da de er fortløbne skrevede linje for linje
Et vigtigt punkt er at teste hvilke metoder, der så virker, hvis ovenstående ikke virker

Udviklingen skal ske ud fra samme designfilosofi som i andre projekter jeg har haft, med overskuelig filstruktur, udviklingsstages med separat kontekst mm. (har glemt navnet, det skal agent finde frem)
Der skal meget testing til at finde ud af hvilke metoder, der virker bedst til at transskribere andensiderne. Derfor skal der laves en testfilstruktur, hvor forskellige metoder kan testes og evalueres.
Der er adgang til håndtransskriberede sider, som kan bruges til at teste forskellige metoder og validere op imod. Jeg har ikke udforsket transskriptionerne endnu, men de er øjensynligt opmærket med tekst, linjeskift, "[?]" for usikkerhed, noget beskrivelse efter linjen i [], men det er ikke sikkert at der er en helt hård standard for hvordan det er gjort. De ligger under "<kilderod>\PID-scapes and Blegdam Patient journals\Patient journals\Manual transcriptions", særligt i undermappen deaths

For mig at se er udviklingsarbejdet objektorienteret men jeg arbejder også godt, med en målsætning om testdriven design.

Denne metode skal kunne plugge ind i det dashboard/web app som min kollega har lavet i samme GH repo som ovenfor. Derfor ikke noget meget nyt design - det skal ligesom kunne loades ind som en metode
Vi kender pt. destinationen for alle forsider i billedmapperne med alle scanninger af alle patientjournaler, så jeg tror ikke der skal meget til at finde ud af hvor andensiderne er

Konkrete faldgruber allerede nu: 
En side er et opslag i et indbundet bind af patientjournaler, hvor man også kan risikere at se noget af tidligere sider, næste side, mm. Her skal vi finde ud af om vi skal bruge computervision, prompting, klassifikation, eller det hele sammen eller noget helt andet.
Linjer kan være overstregede, hvilket mange ML modeller, LLM og andre transformermodeller ofte kan se igennem. Erfaring viser at det kan være utilstrækkeligt at bruge en prompt, der beder modellen om at ignorere overstregninger. Det kan være en udvej at oprette et separat prompt, der beder om at gøre opmærksom på hvad der kan være overstreget, men der skal man selvfølgelig ramme et sweetspot. Testing nødvendigt.
Andre problemer kan være mærkelig tegnsætning, superscript, subscript, forkortelser. For mig at se er det bedst at beholde prompts så simple som muligt, hvorfor man ikke skal bruge prompting som første udvej