import os, re
import streamlit as st
from openai import OpenAI
from topics import TOPICS

st.set_page_config(page_title="KI-Lehrling", page_icon="🧠", layout="centered")

CORE_PROMPT = '''
Du bist ein adaptiver KI-Lehrling.

ZIEL:
Du hilfst nicht maximal, sondern genau so viel wie nötig.
Dein Ziel ist zunehmende Selbstständigkeit der lernenden Person.

REGELN:
- Verlange zuerst einen eigenen Denk- oder Lösungsansatz.
- Gib nicht sofort vollständige Lösungen.
- Stelle möglichst zuerst eine passende Rückfrage.
- Passe die Hilfe an die konkrete Kompetenz im aktuell gewählten Thema an.
- Übertrage Einschätzungen NICHT auf andere Themen.
- Vergib keine Schulnoten.
- Verwende ausschließlich das aktuell aktive Fachmodul als Lernkontext.
- Stelle immer nur EINE Frage oder EINEN Arbeitsauftrag auf einmal.

UNTERSTÜTZUNGSSTUFEN:
0 – Selbstständig: nur prüfen.
1 – Rückfrage: gezielte Frage.
2 – Hinweis: kleiner fachlicher Hinweis.
3 – Führung: in Teilschritte zerlegen.
4 – Erklärung: Grundlage knapp erklären und danach ähnliche Aufgabe selbst lösen lassen.

ADAPTIVE STEUERUNG:
Beurteile intern nach jeder Antwort:
- fachliche Richtigkeit 0–2
- Begründung 0–2
- Selbstständigkeit 0–2

5–6: Schwierigkeit erhöhen, Unterstützung reduzieren.
3–4: gleiches Niveau, kurze Rückfrage oder Hinweis.
0–2: einfacher Zwischenfall, Unterstützung erhöhen.
Zeige diese Punkte NICHT an.

DIDAKTIK:
- Bei Fehlern zuerst zum Fehler führen.
- Richtige Antworten gelegentlich begründen lassen.
- Sichere Bereiche nicht unnötig wiederholen.
- Bei fehlendem Grundverständnis zurückgehen.
- Kurze, realistische Praxisfälle verwenden.
- Nicht reflexartig richtig/falsch sagen.
- Fachlich, knapp und verständlich formulieren.

Wenn die lernende Person "LEHRLINGSKARTE" schreibt, erstelle ausschließlich:

LEHRLINGSKARTE – [AKTUELLES THEMA]

Kompetenzen:
- [Kompetenz]: sicher / teilweise sicher / Unterstützung notwendig

Das beherrsche ich besonders gut:
- ...

Das verwechsle ich noch:
- ...

Hier brauche ich noch Unterstützung:
- ...

Meine momentane Unterstützungsstufe in diesem Thema:
0 / 1 / 2 / 3 / 4

Meine nächste sinnvolle Herausforderung:
- ...

Die Lehrlingskarte gilt ausschließlich für das aktuell aktive Thema.
'''

def setting(name, default=None):
    return st.secrets[name] if name in st.secrets else os.getenv(name, default)

api_key = setting("API_KEY", "")
base_url = setting("BASE_URL", "https://api.groq.com/openai/v1")
model = setting("MODEL", "openai/gpt-oss-120b")

if not api_key:
    st.error("API_KEY fehlt. Bitte in den Streamlit-Secrets hinterlegen.")
    st.stop()

client = OpenAI(api_key=api_key, base_url=base_url)

def recover(raw):
    for pattern in [
        r'["\\\']response["\\\']\\s*:\\s*["\\\'](.+?)["\\\']\\s*,\\s*["\\\']support_level',
        r'\\\"response\\\"\\s*:\\s*\\\"(.+?)\\\"\\s*,\\s*\\\"support_level',
    ]:
        m = re.search(pattern, raw, re.S)
        if m:
            return m.group(1).replace(r'\\n', '\n').replace(r'\\\"', '"')
    return None

def call_model(messages, max_tokens=900):
    try:
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            tool_choice="none",
        )
        return r.choices[0].message.content
    except Exception as e:
        raw = str(e)
        if "tool_use_failed" in raw:
            x = recover(raw)
            if x:
                return x
        raise RuntimeError("MODEL_CALL_FAILED")

def system_prompt(topic_name):
    t = TOPICS[topic_name]
    competencies = "\n".join(f"- {x}" for x in t["competencies"])
    return f'''{CORE_PROMPT}

AKTIVES FACHMODUL:
{topic_name}

KOMPETENZEN:
{competencies}

{t["module_prompt"]}

WICHTIG:
- Bleibe ausschließlich in diesem Fachmodul.
- Verwende keine Lernstandsannahmen aus anderen Modulen.
- Die Diagnose soll maximal fünf Fragen umfassen.
- Stelle Diagnosefragen EINZELN.
'''

for key, default in [
    ("messages", {}),
    ("started", {}),
    ("user_turns", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default
if "topic" not in st.session_state:
    st.session_state.topic = list(TOPICS.keys())[0]

st.title("🧠 KI-Lehrling")
st.caption("Adaptive Lernbegleitung: so wenig Hilfe wie möglich, so viel wie nötig.")

names = list(TOPICS.keys())
selected = st.selectbox("Themengebiet", names, index=names.index(st.session_state.topic))
st.session_state.topic = selected
t = TOPICS[selected]

st.session_state.messages.setdefault(selected, [])
st.session_state.started.setdefault(selected, False)
st.session_state.user_turns.setdefault(selected, 0)

st.subheader(f'{t["icon"]} {t["subtitle"]}')

turns = st.session_state.user_turns[selected]
if not st.session_state.started[selected]:
    phase, progress = "Noch nicht gestartet", 0.0
elif turns < 5:
    phase, progress = f"Diagnose {min(turns+1,5)}/5", min((turns+1)/5*0.35, 0.35)
else:
    phase, progress = "Adaptive Lernphase", min(0.35+(turns-4)*0.05, 0.9)

st.progress(progress)
st.caption(f"Phase: **{phase}**")

with st.expander("🎯 Kompetenzen dieses Moduls"):
    for c in t["competencies"]:
        st.markdown(f"- {c}")

with st.expander("ℹ️ So arbeitest du"):
    st.markdown('''
- Beantworte jede Frage zuerst **selbstständig**.
- Begründe deine Entscheidung.
- Nutze **Hinweis**, wenn du feststeckst.
- Mit **Lehrlingskarte** erhältst du eine Lernstandszusammenfassung.
- Beim Themenwechsel bleibt der Lernkontext getrennt.
''')

c1, c2 = st.columns(2)
with c1:
    if st.button("▶ Lernstrecke starten", use_container_width=True):
        st.session_state.messages[selected] = []
        st.session_state.user_turns[selected] = 0
        st.session_state.started[selected] = True
        try:
            a = call_model([
                {"role":"system","content":system_prompt(selected)},
                {"role":"user","content":"Starte jetzt. Begrüße mich in einem kurzen Satz und stelle nur die erste Diagnosefrage."}
            ], 500)
            st.session_state.messages[selected].append({"role":"assistant","content":a})
            st.rerun()
        except Exception:
            st.error("Der KI-Lehrling konnte nicht starten. Bitte erneut versuchen.")
with c2:
    if st.button("↻ Dieses Thema neu beginnen", use_container_width=True):
        st.session_state.messages[selected] = []
        st.session_state.user_turns[selected] = 0
        st.session_state.started[selected] = False
        st.rerun()

b1, b2 = st.columns(2)
with b1:
    if st.button("💡 Kleiner Hinweis", use_container_width=True, disabled=not st.session_state.started[selected]):
        st.session_state.messages[selected].append({"role":"user","content":"Ich brauche einen kleinen Hinweis. Erhöhe die Unterstützung nur um EINE Stufe. Gib nicht die vollständige Lösung."})
        try:
            msgs=[{"role":"system","content":system_prompt(selected)}]+st.session_state.messages[selected]
            a=call_model(msgs,500)
            st.session_state.messages[selected].append({"role":"assistant","content":a})
            st.rerun()
        except Exception:
            st.error("Hinweis konnte gerade nicht erzeugt werden.")
with b2:
    if st.button("📋 Lehrlingskarte erstellen", use_container_width=True, disabled=not st.session_state.started[selected]):
        st.session_state.messages[selected].append({"role":"user","content":"LEHRLINGSKARTE"})
        try:
            msgs=[{"role":"system","content":system_prompt(selected)}]+st.session_state.messages[selected]
            a=call_model(msgs,900)
            st.session_state.messages[selected].append({"role":"assistant","content":a})
            st.rerun()
        except Exception:
            st.error("Lehrlingskarte konnte gerade nicht erstellt werden.")

st.divider()

for msg in st.session_state.messages[selected]:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
        st.markdown(msg["content"])

if st.session_state.started[selected]:
    prompt=st.chat_input(f"Deine Antwort – {selected}")
    if prompt:
        st.session_state.messages[selected].append({"role":"user","content":prompt})
        st.session_state.user_turns[selected]+=1
        with st.chat_message("user",avatar="👤"):
            st.markdown(prompt)
        msgs=[{"role":"system","content":system_prompt(selected)}]+st.session_state.messages[selected]
        with st.chat_message("assistant",avatar="🤖"):
            with st.spinner("KI-Lehrling denkt …"):
                try:
                    a=call_model(msgs)
                    st.markdown(a)
                    st.session_state.messages[selected].append({"role":"assistant","content":a})
                except Exception:
                    st.error("Der KI-Lehrling konnte gerade nicht antworten. Bitte nochmals versuchen.")
else:
    st.info("Klicke auf **Lernstrecke starten**.")

st.divider()
st.caption("Version 2.1 · Themen werden getrennt geführt. Keine sensiblen personenbezogenen Daten eingeben.")
