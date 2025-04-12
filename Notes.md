# Person feature detection
Person feature detection (entstanden aus dem Modul Computer Vision letztes Semester) sind unterschiedliche KI-Modelle die Eigenschaften im Gesicht von Personen erkennen sollen, sowohl auf Bildern als auch in einem live Video.

Bisher erkennbare Eigenschaften sind: Geschlecht, (Herkunft), Haarfarbe, Bart oder kein Bart, Brille oder keine Brille

Das besondere dabei ist dass für die Eigenschaft Geschlecht ein KI-Modell selber geschrieben/"entdeckt" wurde.


### Ziele/Optimierungen für IPROF
1. Bild Format (JPG) anschauen, ob das so gut für die Eingabe ist und auch später für die Data Augmentierung

2. Live video erkennung präziser machen, indem ein Face detection Model im Hintergrund mitläuft und erst im live feed ein Gesicht extrahiert und dieses dann im Anschluss als Input Bild für die KI-Modelle zum vorhersagen benutzt.
    - Aktuell wird nämlich im live video das gesamte Bild als Input benutzt und das führt zu Problemen, wie das die Modelle keine klaren und sicheren Vorhersagen treffen können.

3. Verschiedene Data Augmentierungs Arten ausprobieren und evaluieren. Kann man das ganze eventuell Automatisieren (Pipeline)?
    - Genauigkeiten der Modelle sollen hoffentlich dadurch besser werden
    - Am Ende könnte man es noch vergleichen?


### Fragen