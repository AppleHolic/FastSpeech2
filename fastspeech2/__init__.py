import os


# import all
def __import_models():
    py_files = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
    for f in py_files:
        if '__init__.py' == f:
            continue
        else:
            file_name = f.split('.')[0]
            exec('from fastspeech2.models import {}'.format(file_name))


__import_models()
