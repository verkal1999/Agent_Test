# Vergleichstabellen TC-003 und TC-004

Basis sind die vorhandenen JSON-Dateien in `Evaluation/results`. Die Gegenueberstellung erfolgt paarweise nach gleichem TestCase sowie gleichem Provider/Modell. Fuer `TC-003/Groq` liegt nur ein KG-Ergebnis vor; deshalb ist die RAG-Spalte dort als nicht verfuegbar markiert.

## TC-003 / Anthropic / Claude Sonnet 4.6

| Kriterium | ExcH-/KG-Agent | RAG-Agent |
| --- | --- | --- |
| Grundausrichtung der Antwort | Rekonstruiert den Hardware-Fehlerpfad kausal von `GVL_HRL.HRL_LS_aussen` ueber `WorkpieceAtPickupPosition`, `VSG_ErrorDetected`, `ErrorDetectedRaw` und `rtError.Q` bis `OPCUA.TriggerD2`; bezieht `HRL_NMethod_Auslagern`, Nachbarsensor und Simulation mit ein. | Interpretiert das Fehlerbild primaer als Wissensgraph-/Failure-Mode-Luecke und als Problem der fehlenden POU `VSG_AS_CompressorControl`; bleibt von der eigentlichen Sensorursache weg. |
| Benennt das setzende POU | Ja, `VSG_DiagnosisHandler` setzt `OPCUA.TriggerD2`. | Nein. Es werden `SecuringWorkpiece` und `VSG_AS_CompressorControl` diskutiert, aber nicht das setzende POU fuer den Trigger. |
| Benennt die eigentliche Ursache | Ja, `GVL_HRL.HRL_LS_aussen = FALSE`; interpretiert als fehlendes/verschobenes Werkstueck oder Sensordefekt und nennt zusaetzlich `GVL_HRL_Sim.HRL_LS_aussen`. | Nein, die eigentliche Root Cause am Sensor wird verfehlt; stattdessen Fokus auf fehlende Failure Modes und fehlende POU-Implementierung. |
| Kausalkette ueber mehrere FBs | Ja, detailliert ueber `VSG_AS_CompressorControl` -> `VSG_DiagnosisHandler`, inkl. Zwischenvariablen `WorkpieceAtPickupPosition`, `VSG_ErrorDetected`, `ErrorDetectedRaw`, `rtError.Q`. | Nur eingeschraenkt; es werden strukturelle Analogien wie `SecuringWorkpiece` und `AxisControl_Encoder` genannt, aber keine belastbare Kette bis zum Sensor konstruiert. |
| Trennung zwischen Trigger-Bedingung und Ursache | Ja, Trigger (`rtError.Q` / `OPCUA.TriggerD2`) und upstream cause (`GVL_HRL.HRL_LS_aussen = FALSE`) werden sauber getrennt. | Nein; unmittelbares Fehlerbild "no failure modes" und eigentliche Ursache werden weitgehend vermischt. |
| Massnahmen / Empfehlungen | Ja, konkrete Hardware-Diagnose: Sensor-LED/Klemme, Werkstueckposition, Verkabelung, Online-Wert, Simulation und Nachbarsensor pruefen. | Nur allgemeine Kontext- und Konfigurationspruefungen wie Export, `OperatingMode`, `MethodCall` und KG/FMEA-Eintraege; keine zielgerichtete Hardware-Diagnose. |
| root_cause_found | true | false |
| judge_verdict | correct | incorrect |
| judge_summary | Die Modellantwort identifiziert korrekt, dass `GVL_HRL.HRL_LS_aussen` (`FALSE`) die Ursache ist, repraesentiert einen Sensor, und leitet plausible Ursachen sowie konkrete Diagnoseschritte ab. Das Simulations-Mapping wird explizit erkannt und hervorgehoben. | Die Modellantwort identifiziert korrekt den fehlenden Wissensgraph-Eintrag als unmittelbare Ausloesebedingung, verfehlt aber die eigentliche Root Cause laut Ground Truth vollstaendig: ein fehlendes Werkstueck am Sensor `GVL_HRL.HRL_LS_aussen`. |
| Laufzeit (s) | 77.453 | 70.906 |
| Tokenverbrauch | 9738 (5642 Prompt / 4096 Completion) | 48787 (45154 Prompt / 3633 Completion) |
| Kosten (USD) | 0.078366 | 0.189957 |
| Stages | 3 | 4 |
| Auffaellige Notiz | Sehr ausfuehrlich; erkennt das Simulations-Mapping explizit; die konfigurierte Frage wurde wegen KG-UI-Emulation ignoriert. | `GVL_HRL.HRL_LS_aussen = false` war im Snapshot sichtbar, wurde aber nicht als Ursache interpretiert; die konfigurierte Frage wurde wegen RAG-UI-Emulation ignoriert. |

Kurze Interpretation: In diesem Paar ist der qualitative Abstand am groessten. Der KG-Agent rekonstruiert die technische Kette sauber bis zum Sensorsignal und ist trotz hoher Kosten klar praeziser als der RAG-Agent, der das Problem auf eine Modellierungsluecke im Wissensgraphen verschiebt und dafuer sogar deutlich mehr Tokens verbraucht.

## TC-003 / OpenAI / GPT-4o mini

| Kriterium | ExcH-/KG-Agent | RAG-Agent |
| --- | --- | --- |
| Grundausrichtung der Antwort | Rekonstruiert den Fehlerpfad deterministisch vom Sensor `GVL_HRL.HRL_LS_aussen` bis zur Ausloesung von `OPCUA.TriggerD2` und ergaenzt danach eine Hardware-Diagnose. | Bleibt zunaechst auf der Ebene von `VSG_SkillSet`, Kompressorsteuerung und in der Folgeanalyse auf `AxisControl_Encoder`; die eigentliche Sensorursache wird nicht herausgearbeitet. |
| Benennt das setzende POU | Ja, `VSG_DiagnosisHandler`. | Nein. Genannt werden eher `VSG_SkillSet` und spaeter `AxisControl_Encoder`. |
| Benennt die eigentliche Ursache | Ja, fehlendes Werkstueck bzw. Sensorproblem an `GVL_HRL.HRL_LS_aussen`; die Moeglichkeit einer Simulation wird mitgefuehrt. | Nein, die Antwort fokussiert Motor-/Kompressorlogik statt das Sensorsignal `GVL_HRL.HRL_LS_aussen`. |
| Kausalkette ueber mehrere FBs | Ja, `GVL_HRL.HRL_LS_aussen` -> `WorkpieceAtPickupPosition` -> `VSG_ErrorDetected` -> `ErrorDetectedRaw` -> `rtError.Q` -> `OPCUA.TriggerD2`. | Nein, nur spekulative Verweise auf `AxisControl_Encoder` und fehlende Laufzeitwerte; keine belastbare Kette bis zur Ursache. |
| Trennung zwischen Trigger-Bedingung und Ursache | Ja, unmittelbarer Trigger und upstream cause werden getrennt dargestellt. | Nein, Trigger, vermutete Kompressorstoerung und moegliche Folgeeffekte werden nicht sauber voneinander getrennt. |
| Massnahmen / Empfehlungen | Ja, konkrete Hardware-Checks am Sensor, Werkstueck und ggf. Simulationsquelle. | Nur allgemeine naechste Checks zu `ResetMethodCall`, `EmergencyStopSignal` und weiteren Laufzeitwerten; keine zielgerichtete Sensor-Diagnose. |
| root_cause_found | true | false |
| judge_verdict | correct | incorrect |
| judge_summary | Die Modellantwort identifiziert korrekt die Ursache des Fehlers als ein fehlendes Werkstueck oder einen defekten Sensor und schlaegt entsprechende Diagnose- und Behandlungsschritte vor. | Die Modellantwort identifiziert nicht die tatsaechliche Ursache des Fehlers, die in einem fehlenden Werkstueck an `GVL.HRL_LS_aussen` liegt. |
| Laufzeit (s) | 34.844 | 48.203 |
| Tokenverbrauch | 5156 (4092 Prompt / 1064 Completion) | 34760 (33848 Prompt / 912 Completion) |
| Kosten (USD) | 0.001252 | 0.005624 |
| Stages | 3 | 4 |
| Auffaellige Notiz | Beruecksichtigt auch die Moeglichkeit einer Simulation und bietet eine detaillierte Analyse der Signalpfade; KG-UI-Emulation ignorierte die konfigurierte Frage. | Der Sensor `GVL.HRL_LS_aussen` und die Moeglichkeit eines fehlenden Werkstuecks bleiben unberuecksichtigt; RAG-UI-Emulation ignorierte die konfigurierte Frage. |

Kurze Interpretation: Das OpenAI-Paar zeigt den fuer die Arbeit wahrscheinlich staerksten Effizienzkontrast. Der KG-Agent ist hier nicht nur inhaltlich korrekt, sondern auch massiv sparsamer bei Tokens, Laufzeit und Kosten als der RAG-Agent, der sich in eine falsche Motor-/Encoder-Richtung verrennt.

## TC-003 / Together / Llama 3.3 70B Instruct Turbo

| Kriterium | ExcH-/KG-Agent | RAG-Agent |
| --- | --- | --- |
| Grundausrichtung der Antwort | Rekonstruiert den Fehlerpfad ebenfalls von `GVL_HRL.HRL_LS_aussen` Richtung `OPCUA.TriggerD2`, bleibt in der Diagnose aber etwas knapper und weniger differenziert als die staerkeren KG-Laeufe. | Bleibt auf der Ebene fehlender Fehlermodi fuer `VSG_AS_CompressorControl` und wechselt danach spekulativ zu `AxisControl_Encoder`; die eigentliche Sensorursache wird nicht getroffen. |
| Benennt das setzende POU | Ja, `VSG_DiagnosisHandler`. | Nein. Als zentrale POU wird eher `AxisControl_Encoder` genannt. |
| Benennt die eigentliche Ursache | Ja, `GVL_HRL.HRL_LS_aussen = FALSE`; Ursache wird als Sensor-/Werkstueckproblem erkannt, aber weniger differenziert zwischen Sensordefekt und Werkstueckentfernung. | Nein, es bleibt bei Fehlermodus-/Faehigkeitsinterpretation und allgemeinen Hinweisen auf `VSG_ErrorDetected`. |
| Kausalkette ueber mehrere FBs | Ja, die zentrale Signalfolge ueber `WorkpieceAtPickupPosition`, `VSG_ErrorDetected`, `ErrorDetectedRaw`, `rtError.Q` und `OPCUA.TriggerD2` wird rekonstruiert. | Nein, keine belastbare kausale Kette bis zum Sensorsignal; stattdessen allgemeine Vermutungen zu FlipFlops, TOFs und fehlenden Snapshot-Werten. |
| Trennung zwischen Trigger-Bedingung und Ursache | Ueberwiegend ja, aber weniger explizit ausdifferenziert als bei Anthropic/OpenAI-KG. | Nein, Trigger, Fehlermodus und eigentliche Ursache werden nicht sauber getrennt. |
| Massnahmen / Empfehlungen | Ja, grundsaetzlich hardwarebezogene Prufschritte; allerdings weniger praezise und ohne expliziten Simulationsbezug. | Nur allgemeine naechste Schritte und weitere Datenanforderungen; keine zielgerichtete Hardware-Diagnose am Sensorpfad. |
| root_cause_found | true | false |
| judge_verdict | partially_correct | incorrect |
| judge_summary | Die Modellantwort identifiziert den Hardware-Endpunkt `GVL_HRL.HRL_LS_aussen` als Ursache fuer den Fehler, aber die Analyse koennte tiefer gehen und explizit zwischen Sensorfehler und Werkstueckentfernung unterscheiden. | Die Modellantwort identifiziert nicht die korrekte Ursache des Fehlers, naemlich das fehlende Werkstueck an `GVL.HRL_LS_aussen`. |
| Laufzeit (s) | 40.515 | 46.860 |
| Tokenverbrauch | 5285 (4327 Prompt / 958 Completion) | 36058 (35123 Prompt / 935 Completion) |
| Kosten (USD) | 0.004651 | 0.031731 |
| Stages | 3 | 4 |
| Auffaellige Notiz | Solider KG-Lauf, aber ohne explizite Erwaehnung der Simulation und mit geringerer Differenzierung; KG-UI-Emulation ignorierte die konfigurierte Frage. | Die Antwort signalisiert indirekt weiteren Informationsbedarf, obwohl der relevante Kontext bereits vorhanden war; RAG-UI-Emulation ignorierte die konfigurierte Frage. |

Kurze Interpretation: Auch hier liegt der KG-Agent klar vorne, allerdings nicht ganz so dominant wie bei Anthropic oder OpenAI. Das Together-KG-Modell trifft die Root Cause grundsaetzlich, bleibt aber analytisch etwas flacher; der RAG-Lauf verfehlt die Ursache dennoch klar und verbraucht ein Mehrfaches an Tokens.

## TC-003 / Groq / Llama 3.3 70B Versatile (nur KG-Ergebnis vorhanden)

| Kriterium | ExcH-/KG-Agent | RAG-Agent |
| --- | --- | --- |
| Grundausrichtung der Antwort | Rekonstruiert den Pfad von `GVL_HRL.HRL_LS_aussen` bis `OPCUA.TriggerD2` und ordnet den Fehler als hardwareseitigen Sensor-/Werkstueckpfad ein. | Kein korrespondierendes RAG-Ergebnis in `Evaluation/results` vorhanden. |
| Benennt das setzende POU | Ja, `VSG_DiagnosisHandler`. | n.v. |
| Benennt die eigentliche Ursache | Ja, `GVL_HRL.HRL_LS_aussen` als Hardware-Endpunkt; allerdings noch nicht vollstaendig ausdifferenziert. | n.v. |
| Kausalkette ueber mehrere FBs | Ja, mit den zentralen Zwischenstationen `WorkpieceAtPickupPosition`, `VSG_ErrorDetected`, `ErrorDetectedRaw`, `rtError.Q`, `OPCUA.TriggerD2`. | n.v. |
| Trennung zwischen Trigger-Bedingung und Ursache | Ueberwiegend ja. | n.v. |
| Massnahmen / Empfehlungen | Grundsaetzlich vorhanden, aber noch nicht detailliert genug und ohne umfassende Einordnung der Simulation. | n.v. |
| root_cause_found | true | n.v. |
| judge_verdict | partially_correct | n.v. |
| judge_summary | Die Modellantwort identifiziert den Hardware-Endpunkt `GVL_HRL.HRL_LS_aussen` als Ursache fuer den Fehler, aber die Analyse ist teilweise unvollstaendig und enthaelt einige unklare Punkte. | Kein korrespondierendes RAG-Ergebnis in `Evaluation/results` vorhanden. |
| Laufzeit (s) | 18.640 | n.v. |
| Tokenverbrauch | 5239 (4327 Prompt / 912 Completion) | n.v. |
| Kosten (USD) | 0.000000 | n.v. |
| Stages | 3 | n.v. |
| Auffaellige Notiz | Root Cause grundsaetzlich gefunden, aber Diagnose und Behandlungsschritte koennten detaillierter sein; KG-UI-Emulation ignorierte die konfigurierte Frage. | Fuer `TC-003_rag_groq` liegt nur eine Config, aber kein Resultat unter `Evaluation/results` vor. |

Kurze Interpretation: Der Groq-KG-Lauf ist fuer sich genommen brauchbar und erkennt die zentrale Ursache, bleibt aber hinter den staerkeren KG-Laeufen bei Differenzierung und Handlungstiefe zurueck. Ein direkter KG-vs.-RAG-Vergleich ist hier mit dem aktuellen Datenstand nicht moeglich.

## TC-004 / OpenAI / GPT-4o

| Kriterium | ExcH-/KG-Agent | RAG-Agent |
| --- | --- | --- |
| Grundausrichtung der Antwort | Rekonstruiert den Hardware-Fehlerpfad erneut deterministisch vom Sensorsignal `GVL_HRL.HRL_LS_aussen` bis `OPCUA.TriggerD2` und leitet daraus eine Hardware-Diagnose ab. | Bleibt auf der Ebene fehlender Failure Modes, `VSG_ErrorCode = 1001` und spaeter allgemeiner `AxisControl_Encoder`-Logik; die eigentliche Sensorursache wird verfehlt. |
| Benennt das setzende POU | Ja, `VSG_DiagnosisHandler`. | Nein. |
| Benennt die eigentliche Ursache | Ja, Problem am Sensor `GVL_HRL.HRL_LS_aussen`, plausibel als fehlendes Werkstueck oder Sensordefekt; Simulation wird mitbetrachtet. | Nein, die Antwort fokussiert auf Fehlermodus-/Fehlercodeebene statt auf das Sensorsignal und das fehlende Werkstueck. |
| Kausalkette ueber mehrere FBs | Ja, `GVL_HRL.HRL_LS_aussen` -> `WorkpieceAtPickupPosition` -> `VSG_ErrorDetected` -> `ErrorDetectedRaw` -> `rtError.Q` -> `OPCUA.TriggerD2`. | Nein, keine tragfaehige Kausalkette bis zur Root Cause; spaetere Verlagerung auf `AxisControl_Encoder` bleibt spekulativ. |
| Trennung zwischen Trigger-Bedingung und Ursache | Ja, unmittelbarer Trigger und upstream cause werden sauber unterschieden. | Nein, Fehlercode, fehlende Fehlermodi und eigentliche Ursache werden nicht getrennt. |
| Massnahmen / Empfehlungen | Ja, konkrete Prufschritte an Sensor, Werkstueck, Verkabelung und Simulationsquelle. | Nur allgemeine Pruefung der Implementierung bzw. weiterer Laufzeitwerte; keine zielgerichtete Hardware-Diagnose. |
| root_cause_found | true | false |
| judge_verdict | correct | incorrect |
| judge_summary | Die Modellantwort identifiziert korrekt die Ursache des Fehlers als ein Problem mit dem Sensor `GVL_HRL.HRL_LS_aussen` und schlaegt plausible Diagnose- und Behandlungsschritte vor. | Die Modellantwort identifiziert nicht die tatsaechliche Ursache des Fehlers, die in der Ground Truth angegeben ist. |
| Laufzeit (s) | 20.969 | 35.328 |
| Tokenverbrauch | 4829 (4092 Prompt / 737 Completion) | 34598 (33848 Prompt / 750 Completion) |
| Kosten (USD) | 0.017600 | 0.092120 |
| Stages | 3 | 4 |
| Auffaellige Notiz | Die Antwort beruecksichtigt die Moeglichkeit eines Simulationsproblems, was in der Ground Truth explizit als Pluspunkt genannt wird; KG-UI-Emulation ignorierte die konfigurierte Frage. | Der Sensor `GVL_HRL.HRL_LS_aussen` haette als Ursache identifiziert werden muessen; stattdessen bleibt die Antwort auf Fehlermodus-/Fehlercodeebene; RAG-UI-Emulation ignorierte die konfigurierte Frage. |

Kurze Interpretation: `TC-004` bestaetigt das bereits in `TC-003` sichtbare Muster. Der KG-Agent liefert die technisch richtige, kausale und handlungsorientierte Analyse bei deutlich geringerem Ressourcenverbrauch, waehrend der RAG-Agent trotz hoher Tokenlast am eigentlichen Sensorsignal vorbeilaeuft.
