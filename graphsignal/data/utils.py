


def obj_to_dict(obj, level=0):
    if level >= 10:
        return
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: obj_to_dict(v, level=level+1) for k, v in obj.items()}
    elif isinstance(obj, (list, set, tuple)):
        return [obj_to_dict(e, level=level+1) for e in obj]
    elif hasattr(obj, '__dict__'):
        return obj_to_dict(vars(obj), level=level+1)
    else:
        return str(obj)