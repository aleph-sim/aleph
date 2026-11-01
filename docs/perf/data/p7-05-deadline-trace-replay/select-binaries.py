from pathlib import Path
import json
root=Path('/tmp/aleph-547-508c4d0')
for source,kind,out in [('example-artifacts.jsonl','example','example-executable.txt'),('bench-artifacts.jsonl','bench','bench-executable.txt')]:
 found=[]
 for line in (root/source).read_text().splitlines():
  m=json.loads(line)
  if m.get('reason')=='compiler-artifact' and m['target']['name']=='deadline_replay' and kind in m['target']['kind'] and m.get('executable'):found.append(m['executable'])
 assert len(found)==1,(source,found)
 (root/out).write_text(found[0])
