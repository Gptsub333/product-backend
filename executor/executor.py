import importlib

def run_pipeline(module_list):
    module_map = {m['id']: m for m in module_list}
    current = module_list[0]
    result = None

    while current:
        module_type = current['type']
        module = importlib.import_module(f"modules.{module_type}")
        result = module.run({**current['params'], **(result or {})})
        next_id = current.get("next")
        current = module_map.get(next_id) if next_id else None

    return result
