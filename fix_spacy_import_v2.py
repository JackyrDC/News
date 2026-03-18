import json

with open('NEWs.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell.get('source', []):
            if 'import os' in line and 'seaborn as sns' in line and 'spacy' in line:
                line = line.replace(', spacy', '')
                new_source.append(line)
                new_source.append("try:\n")
                new_source.append("    import spacy\n")
                new_source.append("except Exception:\n")
                new_source.append("    spacy = None\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('NEWs.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
