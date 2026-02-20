# agent_results

Ablage fuer Laufzeitergebnisse des Agents und der Streamlit-UI.

## Dateitypen
- `*_event.json`: Eingangsereignis (normalisiert oder roh).
- `*_result.json`: Agent-Ergebnis fuer das jeweilige Ereignis.
- `streamlit_*_<corrId>/chatBot_verlauf.json`: Sessionprotokoll inkl. Chatverlauf.

## Hinweis
- Diese Daten sind Laufzeitartefakte und typischerweise nicht die Quelle der Kernlogik.
