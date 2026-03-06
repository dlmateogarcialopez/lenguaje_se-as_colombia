import json

def preview_json_schema(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = ""
        for _ in range(200):
            try:
                line = next(f)
                if '"' in line:
                    print(line.strip('\n'))
            except StopIteration:
                break

preview_json_schema(r'd:\LSC\LSCS45\sample.json')
