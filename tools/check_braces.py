import sys

PAIRS = {'}': '{', ')': '(', ']': '['}
OPEN = set('{([')


def check(path):
    src = open(path, encoding='utf-8', errors='replace').read()
    depth = {'{': 0, '(': 0, '[': 0}
    i = 0
    n = len(src)
    line = 1
    state = 'code'          # code | line_comment | block_comment | string | char | raw_string
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ''
        if c == '\n':
            line += 1
            if state == 'line_comment':
                state = 'code'
            i += 1
            continue
        if state == 'code':
            if c == '/' and nxt == '/':
                state = 'line_comment'; i += 2; continue
            if c == '/' and nxt == '*':
                state = 'block_comment'; i += 2; continue
            if c == '"':
                state = 'string'; i += 1; continue
            if c == "'":
                state = 'char'; i += 1; continue
            if c in OPEN:
                depth[c] += 1
            elif c in PAIRS:
                o = PAIRS[c]
                depth[o] -= 1
                if depth[o] < 0:
                    return (False, line, 'negative %s' % o)
        elif state == 'line_comment':
            pass
        elif state == 'block_comment':
            if c == '*' and nxt == '/':
                state = 'code'; i += 2; continue
        elif state in ('string', 'char'):
            if c == '\\':
                i += 2; continue
            if (state == 'string' and c == '"') or (state == 'char' and c == "'"):
                state = 'code'
        i += 1
    if state in ('string', 'char'):
        return (False, line, 'unterminated literal')
    bad = [k for k, v in depth.items() if v != 0]
    return (not bad, line, 'depth %s' % depth)


for p in sys.argv[1:]:
    ok, line, info = check(p)
    print('%-40s %s  %s' % (p, 'OK' if ok else 'FAIL', info))
    if not ok:
        sys.exit(1)
