import ast
import json
import re

def _is_apostrophe(s: str, i: int) -> bool:
        """True iff s[i] is a literal apostrophe between alphanumerics: e.g., Doe's / it's."""
        return 0 < i < len(s) - 1 and s[i] == "'" and s[i-1].isalnum() and s[i+1].isalnum()

def _to_valid_json_from_pythonish(s: str) -> str:
    out = []
    i, n = 0, len(s)
    in_str = False
    quote = None
    started_escaped = False
    in_inner = False
    str_before_ns = None  

    def next_non_space(idx: int) -> str:
        j = idx
        while j < n and s[j].isspace():
            j += 1
        return s[j] if j < n else ''

    def prev_non_space(idx: int) -> str:
        j = idx
        while j >= 0 and s[j].isspace():
            j -= 1
        return s[j] if j >= 0 else ''

    def peek(idx: int) -> str:
        return s[idx] if idx < n else ''

    def is_outer_value_terminator(ch: str) -> bool:
        return ch == '' or ch in (',', '}', ']')

    def is_key_colon_terminator(next_ns: str) -> bool:
        return next_ns == ':' and str_before_ns in ('{', ',')

    def is_inner_open_candidate(idx: int) -> bool:
        """Open inner quotes only when the ' is likely starting a quoted token/phrase."""
        prevc = s[idx - 1] if idx > 0 else ''
        nxt1  = peek(idx + 1)
        return (not prevc.isalnum()) and (nxt1 and (nxt1.isalnum() or nxt1 in '._-/$'))

    while i < n:
        ch = s[i]

        if not in_str:
            # Escaped opener outside a string -> start a new string
            if ch == '\\' and i + 1 < n and s[i + 1] in ("'", '"'):
                in_str = True
                quote = s[i + 1]
                started_escaped = True
                str_before_ns = prev_non_space(i - 1)
                out.append('"')
                i += 2
                continue

            if ch in ("'", '"'):
                in_str = True
                quote = ch
                started_escaped = False
                str_before_ns = prev_non_space(i - 1)
                out.append('"')
                i += 1
                continue

            out.append(ch); i += 1; continue

        if ch == '\\':
            if i + 1 < n:
                nxt = s[i + 1]
                if nxt in ("'", '"') and nxt == quote and started_escaped:
                    nxt_ns = next_non_space(i + 2)
                    if is_outer_value_terminator(nxt_ns) or is_key_colon_terminator(nxt_ns):
                        out.append('"')
                        in_str = False
                        quote = None
                        started_escaped = False
                        str_before_ns = None
                        i += 2
                        continue

                if nxt == '"':
                    out.append('\\"'); i += 2; continue
                if nxt == quote:
                    out.append('\\"' if nxt == '"' else nxt); i += 2; continue
                if nxt in ['\\', 'n', 'r', 't', 'b', 'f', '/']:
                    out.append('\\' + nxt); i += 2; continue
                out.append(nxt); i += 2; continue

            i += 1; continue

        if quote == '"':
            if ch == '"':
                out.append('"'); in_str = False; quote = None; started_escaped = False; str_before_ns = None; i += 1; continue
            out.append('\\"' if ch == '"' else ch); i += 1; continue

        if ch == "'":
            if _is_apostrophe(s, i):
                out.append("'"); i += 1; continue
            if in_inner:
                out.append('\\"'); in_inner = False; i += 1; continue
            if is_inner_open_candidate(i):
                in_inner = True
                out.append('\\"')             
                i += 1
                continue
            nxt_ns = next_non_space(i + 1)
            if is_outer_value_terminator(nxt_ns) or is_key_colon_terminator(nxt_ns):
                out.append('"'); in_str = False; quote = None; started_escaped = False; str_before_ns = None; i += 1; continue
            out.append("'"); i += 1; continue
        if ch == '"':
            out.append('\\"'); i += 1; continue
        out.append(ch); i += 1

    return ''.join(out)


def robust_parse(obs: str) -> dict:
    def _escape_newlines_in_strings(s: str) -> str:
        out, i, n = [], 0, len(s)
        in_str, quote = False, None
        while i < n:
            ch = s[i]
            if not in_str and ch in ("'", '"'):
                in_str, quote = True, ch
                out.append(ch); i += 1; continue
            elif in_str:
                if ch == quote:
                    in_str, quote = False, None
                if ch == '\n':
                    out.append('\\n'); i += 1; continue
            out.append(ch); i += 1
        return ''.join(out)
    
    def _clean_inner_quotes(obj):
        if isinstance(obj, dict):
            return {k: _clean_inner_quotes(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean_inner_quotes(v) for v in obj]
        if isinstance(obj, str):
            return re.sub(
                r"(?:^|[^A-Za-z0-9])'([A-Za-z0-9 _\-:.]+)'(?=[^A-Za-z0-9]|$)",
                lambda m: m.group(0).replace("'" + m.group(1) + "'", '"' + m.group(1) + '"'),
                obj,
            )
        return obj

    obs = obs.strip().strip('"').strip("'")
    obs = _escape_newlines_in_strings(obs)
    raw_obs = obs

    rebuilt = []
    i, n = 0, len(obs)
    in_str, quote = False, None
    while i < n:
        ch = obs[i]
        if not in_str and ch in ("'", '"'):
            in_str, quote = True, ch
            rebuilt.append(ch); i += 1; continue
        elif in_str:
            if quote == "'" and ch == "'" and _is_apostrophe(obs, i):
                rebuilt.append(ch); i += 1; continue

            rebuilt.append(ch)
            if ch == '\\' and i + 1 < n:
                rebuilt.append(obs[i+1]); i += 2; continue
            if ch == quote:
                in_str, quote = False, None
            i += 1; continue
        else:
            if obs.startswith("None", i):
                rebuilt.append("null"); i += 4; continue
            if obs.startswith("True", i):
                rebuilt.append("true"); i += 4; continue
            if obs.startswith("False", i):
                rebuilt.append("false"); i += 5; continue
            rebuilt.append(ch); i += 1; continue
    obs = ''.join(rebuilt)
    try:
        result = ast.literal_eval(obs)
        if isinstance(result, dict): return _clean_inner_quotes(result)
    except Exception: pass

    try:
        json_str = _to_valid_json_from_pythonish(obs)
        result = json.loads(json_str)
        if isinstance(result, dict): return _clean_inner_quotes(result)
    except Exception: pass

    try:
        json_str = raw_obs.replace('\"', '\\"').replace("\\'", "\\'")
        result = json.loads(json_str)
        if isinstance(result, dict): return _clean_inner_quotes(result)
    except Exception: pass

    try:
        result = ast.literal_eval(raw_obs)
        if isinstance(result, list): return _clean_inner_quotes(result)
    except Exception: pass

    try:
        if obs.lstrip().startswith('['):
            try: return _clean_inner_quotes(json.loads(obs)) 
            except Exception:
                py_obs = re.sub(r'(?<!\w)null(?!\w)', 'None', re.sub(r'(?<!\w)true(?!\w)', 'True', re.sub(r'(?<!\w)false(?!\w)', 'False', obs)))
                return _clean_inner_quotes(ast.literal_eval(py_obs))
    except Exception: pass
    return str(raw_obs) 
        