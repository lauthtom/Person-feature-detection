# Person feature detection
Person feature detection (entstanden aus dem Modul Computer Vision letztes Semester) sind unterschiedliche KI-Modelle die Eigenschaften im Gesicht von Personen erkennen sollen, sowohl auf Bildern als auch in einem live Video.

Bisher erkennbare Eigenschaften sind: Geschlecht, (Herkunft), Haarfarbe, Bart oder kein Bart, Brille oder keine Brille

Das besondere dabei ist dass für die Eigenschaft Geschlecht ein KI-Modell selber geschrieben/"entdeckt" wurde.

### Fortschritt
- Dokumentation des alten Stands (Video über die Performance/Genauigkeit über das Live Video)
- Komprimiertes format zum speichern der Modelle, sowie auch paralleles laden der Modelle (Mit parallelem laden: ~30 Sekunden, Ohne parallelem laden: ~4 Minuten)
- Python skript erstellt, welches ein Gesicht aus einem Bild erkennen und extrahiert. Des Weiteren wird dieses Bild aufbereitet und weitergegeben für eine Vorhersage der jeweiligen KI-Modelle
- Bild Format (JPG) anschauen, ob das so gut für die Eingabe ist und auch später für die Data Augmentierung
    - JPG hat einen leichten Qualitäts verlust durch die verlustbehaftete Kompression. PNG wäre hier eine verlustfreiere Alternative, aber dafür sind die Dateien deutlich größer. Für diesen Anwendungsfall ist JPG aber ausreichend.
- Face Detection in einem Live Video testen (Funktioniert - kann auch vorgeführt werden)

### TODO:
- Skript erstellen, welches automatisiert den Datensatz für die einzelnen KI-Modelle vorbereitet (**Erledigt**)
- Skript anpassen, um statische Bilder zu testen (**Erledigt**)
- Das speichern der einzelnen Modelle in den python Notebooks anpassen (saved_models/) (**Erledigt**)
- Verschiedene Data Augmentierungs arten testen um die Performance eventuell nochmal zu verbessern (Data Augmenatation Pipeline?) (**In Bearbeitung**)
- Evaluierungs Datensatz erstellen um später daraus Prozentsätze zu erstellen, wie viel besser oder schlechter das ist (**In Bearbeitung**)
- ⁠Erklären können, wie ein Bild bspw. bei dem Gender Modell verarbeitet wird, also Eingabe Größe etc. (**In Bearbeitung**)
- Eventuell schauen, ob es möglich ist, die Face Detection noch genauer zu machen. Aktuell wird nur ein "padding" Bereich von dem Gesicht extrahiert (könnte immernoch beeinflussbaren Hintergrund enthalen!) (**In Bearbeitung**)
- Wenn noch Zeit ist, eventuell Hyperparameter Tuning betreiben (Wäre interessant zu sehen, ob das noch verbesserungen bringt, oder nicht)
- Es müssen die Modelle jetzt noch über die Data Augmenation Pipeline verbessert werden und dann auch noch im richtigen Verzeichnis 'With_DA_Pipeline' speichern (Aktuell dauert ein Durchlauf mit der Pipeline 14 Minuten, da wir das ganze 10 mal machen, ergibt es eine Gesamtzeit von 2:20h pro Modell)


### Ziele/Optimierungen für IPROF
1. Bild Format (JPG) anschauen, ob das so gut für die Eingabe ist und auch später für die Data Augmentierung (**Erledigt**)

2. Live video erkennung präziser machen, indem ein Face detection Model im Hintergrund mitläuft und erst im live feed ein Gesicht extrahiert und dieses dann im Anschluss als Input Bild für die KI-Modelle zum vorhersagen benutzt. (**Erledigt**)
    - Aktuell wird nämlich im live video das gesamte Bild als Input benutzt und das führt zu Problemen, wie das die Modelle keine klaren und sicheren Vorhersagen treffen können. (**Erledigt**)

3. Verschiedene Data Augmentierungs Arten ausprobieren und evaluieren. Kan man das ganze eventuell automatisieren (Pipeline)?
    - Genauigkeiten der Modelle sollen hoffentlich dadurch besser werden
    - Am Ende könnte man es noch vergleichen?


### Fragen
- Wie kann ich am effektivsten die Performance messen/dokumentieren? Sowohl bei dem Live Video, als auch später bei der Data Augmentierung?