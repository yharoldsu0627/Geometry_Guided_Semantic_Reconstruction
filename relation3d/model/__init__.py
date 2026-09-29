__all__ = ['Relation3D']


def __getattr__(name):
    if name == 'Relation3D':
        from .relation3d import Relation3D
        return Relation3D
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
