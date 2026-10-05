import json

with open('backend/notebooks/01_data_understanding.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'Duplicate text records' in src:
            # Fix: sort by ID_COLUMN instead of TEXT_COLUMN (not in selected cols)
            fixed = src.replace(
                '].sort_values(TEXT_COLUMN)',
                '].sort_values(ID_COLUMN)'
            )
            cell['source'] = [fixed]
            print('Fixed duplicate text cell.')

with open('backend/notebooks/01_data_understanding.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook saved.')
