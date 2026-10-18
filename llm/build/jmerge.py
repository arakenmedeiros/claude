"""Escrita segura (com trava) de JSONs compartilhados entre execuções paralelas: substitui só as chaves dos capítulos processados."""
import json, os, fcntl
def merge_write(path, new, chapters, **kw):
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = json.load(open(path)) if os.path.exists(path) else {}
        cur = {k: v for k, v in cur.items() if int(k.split("-")[0]) not in chapters}
        cur.update({k: v for k, v in new.items() if int(k.split("-")[0]) in chapters})
        json.dump(cur, open(path, "w"), **kw)
