# Auto-generated from Programming_Agent.ipynb
# Contains all code cells in execution order.

# Cell 1
import re
import time
import pythoncom
import win32com.client
from pythoncom import com_error

ST_LANG = 6  # TwinCAT ST

METHOD_HDR = re.compile(r"(?m)^\s*//\s*METHOD\s+(?P<name>\w+)\b.*$")

def clean_st_text(text: str) -> str:
    if not text:
        return ""
    # BOM entfernen, Newlines normalisieren
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    # KG Export Kopfzeile entfernen: "// POU <Name> body"
    text = re.sub(r"(?m)^\s*//\s*POU\s+.*?\s+body\s*\n", "", text)
    return text.strip() + "\n"

def split_blob_into_body_and_methods(st_blob: str):
    st_blob = clean_st_text(st_blob)

    headers = list(METHOD_HDR.finditer(st_blob))
    if not headers:
        return st_blob, {}

    body = st_blob[:headers[0].start()].strip() + "\n"
    methods = {}
    for i, h in enumerate(headers):
        name = h.group("name")
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(st_blob)
        methods[name] = clean_st_text(st_blob[start:end])
    return body, methods

def cast_to(obj, iface: str):
    try:
        return win32com.client.CastTo(obj, iface)
    except Exception:
        return obj

RPC_E_CALL_REJECTED = -2147418111

def com_retry(callable_fn, *args, retries: int = 30, delay: float = 0.3, **kwargs):
    last = None
    for _ in range(retries):
        try:
            return callable_fn(*args, **kwargs)
        except com_error as e:
            last = e
            if getattr(e, "hresult", None) == RPC_E_CALL_REJECTED:
                time.sleep(delay)
                pythoncom.PumpWaitingMessages()
                continue
            raise
    raise last

# Cell 2
VS_PROJECT_KIND_SOLUTION_FOLDER = "{66A26720-8FB5-11D2-AA7E-00C04F688DDE}"

def create_dte():
    prog_ids = [
        "TcXaeShell.DTE.17.0", "VisualStudio.DTE.17.0",
        "TcXaeShell.DTE.16.0", "VisualStudio.DTE.16.0",
        "TcXaeShell.DTE.15.0", "VisualStudio.DTE.15.0",
        "VisualStudio.DTE.14.0", "VisualStudio.DTE.12.0", "VisualStudio.DTE.10.0",
    ]

    last_err = None
    for pid in prog_ids:
        try:
            dte = win32com.client.gencache.EnsureDispatch(pid)
            dte.SuppressUI = True
            try:
                dte.MainWindow.Visible = False
            except Exception:
                pass

            try:
                settings = dte.GetObject("TcAutomationSettings")
                settings.SilentMode = True
            except Exception:
                pass

            return dte
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Kein passender DTE ProgID gefunden. Letzter Fehler: {last_err}")

def _safe_get_project_path(proj):
    for attr in ("FullName", "FileName", "UniqueName"):
        try:
            v = getattr(proj, attr)
            if v:
                return str(v)
        except Exception:
            pass
    return None

def _iter_projects(solution):
    def walk(proj):
        try:
            kind = str(proj.Kind).upper()
        except Exception:
            kind = None

        if kind == VS_PROJECT_KIND_SOLUTION_FOLDER:
            try:
                items = proj.ProjectItems
                for i in range(1, items.Count + 1):
                    try:
                        sub = items.Item(i).SubProject
                        if sub is not None:
                            yield from walk(sub)
                    except Exception:
                        continue
            except Exception:
                print("Warn: Konnte Solution Folder nicht durchlaufen:", getattr(proj, "Name", "?"))
                pass
        else:
            yield proj

    for i in range(1, solution.Projects.Count + 1):
        try:
            p = solution.Projects.Item(i)
            yield from walk(p)
        except Exception:
            continue

def open_solution_get_sysman(solution_path: str):
    dte = create_dte()

    com_retry(dte.Solution.Open, solution_path)

    # Projekte einsammeln, tsproj finden
    tc_project = None
    for p in _iter_projects(dte.Solution):
        ppath = _safe_get_project_path(p)
        if ppath and ppath.lower().endswith(".tsproj"):
            tc_project = p
            break

    if tc_project is None:
        seen = []
        for p in _iter_projects(dte.Solution):
            seen.append((getattr(p, "Name", "?"), _safe_get_project_path(p), getattr(p, "Kind", None)))
        raise RuntimeError("Kein TwinCAT .tsproj gefunden. DTE sieht: " + repr(seen[:30]))

    sysman = tc_project.Object  # ITcSysManager
    return dte, sysman

# Cell 3
def enum_children(tree_item):
    children = []
    try:
        for child in tree_item:
            children.append(child)
    except Exception:
        cnt = int(tree_item.ChildCount)
        for i in range(1, cnt + 1):
            children.append(tree_item.Child(i))
    return children

def list_plc_projects(sysman):
    root = sysman.LookupTreeItem("TIPC")
    out = []
    for child in enum_children(root):
        try:
            plc = child.NestedProject
            out.append((plc.Name, plc.PathName))
        except pythoncom.com_error:
            continue
    return out

def find_first_plc_nested_project(sysman, prefer_name: str = None):
    root = sysman.LookupTreeItem("TIPC")
    candidates = []
    for child in enum_children(root):
        try:
            plc = child.NestedProject
            candidates.append(plc)
        except pythoncom.com_error:
            continue

    if not candidates:
        raise RuntimeError("Kein PLC Projekt unter TIPC gefunden (NestedProject klappt nirgends).")

    if prefer_name:
        for plc in candidates:
            if prefer_name.lower() in (plc.Name or "").lower():
                return plc

    return candidates[0]

def get_pous_tree_path(sysman, prefer_plc_project_name: str = None):
    plc = find_first_plc_nested_project(sysman, prefer_name=prefer_plc_project_name)
    pous_path = f"{plc.PathName}^POUs"
    pous_item = sysman.LookupTreeItem(pous_path)
    return pous_item, pous_path

# Cell 4
def _delete_child_safe(parent_item, child_name: str):
    try:
        parent_item.DeleteChild(child_name)
        return True
    except Exception:
        return False

def _recreate_fb_as_st(pous_item, fb_name: str):
    _delete_child_safe(pous_item, fb_name)
    return pous_item.CreateChild(fb_name, 604, "", ST_LANG)

def _get_impl_lang(tree_item):
    pou = cast_to(tree_item, "ITcPlcPou")
    impl = cast_to(pou, "ITcPlcImplementation")
    lang = getattr(impl, "Language", None)
    try:
        return int(lang) if lang is not None else None
    except Exception:
        return None

def _set_decl_impl(tree_item, decl_text: str, impl_text: str):
    pou = cast_to(tree_item, "ITcPlcPou")
    decl = cast_to(pou, "ITcPlcDeclaration")
    impl = cast_to(pou, "ITcPlcImplementation")

    if hasattr(decl, "DeclarationText"):
        decl.DeclarationText = decl_text
    else:
        tree_item.DeclarationText = decl_text

    if hasattr(impl, "ImplementationText"):
        impl.ImplementationText = impl_text
    else:
        tree_item.ImplementationText = impl_text

def _ensure_methods_container(fb_item):
    # Manche Projekte haben einen Methods Ordner, manche nicht
    try:
        return fb_item.LookupChild("Methods")
    except Exception:
        return None

def upsert_job_fb_and_save(
    solution_path: str,
    fb_name: str,
    fb_decl: str,
    st_blob: str,
    pous_tree_path: str = None,
    prefer_plc_project_name: str = None,
):
    dte, sysman = open_solution_get_sysman(solution_path)

    try:
        if pous_tree_path is None:
            pous_item, pous_tree_path = get_pous_tree_path(sysman, prefer_plc_project_name=prefer_plc_project_name)
        else:
            pous_item = sysman.LookupTreeItem(pous_tree_path)

        fb_path = f"{pous_tree_path}^{fb_name}"

        # FB holen oder erstellen
        try:
            fb_item = sysman.LookupTreeItem(fb_path)
        except Exception:
            fb_item = pous_item.CreateChild(fb_name, 604, "", ST_LANG)

        fb_body_impl, methods = split_blob_into_body_and_methods(st_blob)

        # ST sicherstellen
        lang = _get_impl_lang(fb_item)
        if lang is not None and lang != ST_LANG:
            fb_item = _recreate_fb_as_st(pous_item, fb_name)

        # Schreiben, wenn XML Fehler, dann einmal neu erstellen und erneut versuchen
        try:
            _set_decl_impl(fb_item, fb_decl, fb_body_impl)
        except com_error as e:
            msg = str(e)
            if "System.Xml" in msg or "Ungültige Daten auf Stammebene" in msg:
                fb_item = _recreate_fb_as_st(pous_item, fb_name)
                _set_decl_impl(fb_item, fb_decl, fb_body_impl)
            else:
                raise

        # Methoden anlegen
        methods_parent = _ensure_methods_container(fb_item) or fb_item

        for mname, impl_text in methods.items():
            # Methode holen oder erstellen
            try:
                m_item = methods_parent.LookupChild(mname)
            except Exception:
                m_item = None

            if m_item is None:
                try:
                    m_item = methods_parent.CreateChild(mname, 609, "", ST_LANG)
                except Exception:
                    # Fallback: direkt am FB
                    m_item = fb_item.CreateChild(mname, 609, "", ST_LANG)

            # Nur Implementation setzen, Declaration kannst du später erweitern
            try:
                _set_decl_impl(m_item, "", impl_text)
            except com_error as e:
                msg = str(e)
                if "System.Xml" in msg or "Ungültige Daten auf Stammebene" in msg:
                    raise RuntimeError(f"Methode {mname} erwartet offenbar kein ST Text. Bitte prüfe Sprache/POU Typ.")
                raise

        dte.ExecuteCommand("File.SaveAll")

        # Optional: gib dir zurück, was verwendet wurde
        return {"pous_tree_path": pous_tree_path, "fb_path": fb_path}

    finally:
        try:
            dte.Quit()
        except Exception:
            pass

# Cell 5
SOLUTION = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\TestProjektTwinCATEvents.sln"
FB_NAME = "VSG_AS_VerticalMoveEncoder"

ST_BLOB = """// POU VSG_AS_VerticalMoveEncoder body
HRL_RGB_VerticalMoveWithEncoder(
    MethodCall := MethodCall_VerticalMove,
    EmergencyStopSignal := ESS
);
DO_MovingVerticalVSG_01 := HRL_RGB_VerticalMoveWithEncoder.DigitalOutputControl_01;

// METHOD Abort of VSG_AS_VerticalMoveEncoder
callCounterAbort := callCounterAbort + 1;

// METHOD CheckState of VSG_AS_VerticalMoveEncoder
callCounterCheckState := callCounterCheckState + 1;

// METHOD Start of VSG_AS_VerticalMoveEncoder
callCounterStart := callCounterStart + 1;
"""

FB_DECL = f"""FUNCTION_BLOCK {FB_NAME}
VAR
    (* TODO: Variablen *)
END_VAR
"""

result = upsert_job_fb_and_save(
    solution_path=SOLUTION,
    fb_name=FB_NAME,
    fb_decl=FB_DECL,
    st_blob=ST_BLOB,
    pous_tree_path=None,          # dynamisch
    prefer_plc_project_name=None  # optional: z.B. "PlcGenerated"
)

print("Fertig. Verwendeter POUs Pfad:", result["pous_tree_path"])
print("FB Pfad:", result["fb_path"])

