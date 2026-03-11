import ast, sys
files = ['app/server.py','app/memory_service.py','app/training_service.py','app/ollama_service.py']
ok = True
for f in files:
    try:
        ast.parse(open(f, encoding='utf-8').read())
        print(f'{f}: OK')
    except SyntaxError as e:
        print(f'{f}: ERRO - {e}')
        ok = False
if ok:
    print('\nTODOS OK')
else:
    print('\nERROS ENCONTRADOS')
    sys.exit(1)
