def version(module, version):
    v = module.__version__
    if v != version:
        print(f"{module.__name__} version: {v} ❌ (need {version})")
    else:
        print(f"{module.__name__} version: {v} ✅")
