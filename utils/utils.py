import imp

def import_file(full_path_to_module, name='module.name'):
    module_obj = imp.load_source(name, full_path_to_module)
    return module_obj