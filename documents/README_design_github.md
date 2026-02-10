# design_github.md

**design_github.md** is the GitHub-styled version of **design.md**. It has the same content but with underscores in math escaped as `\_` so that subscripts and superscripts render correctly on GitHub.

**To keep in sync:** After editing design.md, regenerate design_github.md from the project root:

```bash
python3 -c "
import re
with open('documents/design.md') as f: c = f.read()
c = re.sub(r'\$\$.*?\$\$', lambda m: m.group(0).replace('_', r'\\\\_'), c, flags=re.DOTALL)
c = re.sub(r'\$[^\$\n]+\$', lambda m: m.group(0).replace('_', r'\\\\_'), c)
lines = c.split('\n')
out = '\n'.join([lines[0], '', '*GitHub version. Keep in sync with design.md.*', ''] + lines[2:])
with open('documents/design_github.md', 'w') as f: f.write(out)
print('Updated design_github.md')
"
```

Or run the same logic as in the script that was used to create design_github.md initially (escape `_` inside `$...$` and `$$...$$`, then write to design_github.md with the one-line header).
