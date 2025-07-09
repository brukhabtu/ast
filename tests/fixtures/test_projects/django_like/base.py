"""Base model classes."""

from typing import Any, Dict, Type


class Field:
    """Model field descriptor."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", None)
    
    def __set__(self, obj, value):
        setattr(obj, f"_{self.name}", value)
    
    def __set_name__(self, owner, name):
        self.name = name


class ModelMeta(type):
    """Metaclass for models."""
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Collect fields
        cls._fields = {}
        for key, value in namespace.items():
            if isinstance(value, Field):
                cls._fields[key] = value
        
        return cls


class Model(metaclass=ModelMeta):
    """Base model class."""
    
    def __init__(self, **kwargs):
        self.id = None
        for field_name, field in self.__class__._fields.items():
            setattr(self, field_name, kwargs.get(field_name))
    
    def save(self) -> None:
        """Save model to database."""
        print(f"Saving {self.__class__.__name__} instance")
    
    def delete(self) -> None:
        """Delete model from database."""
        print(f"Deleting {self.__class__.__name__} instance")
    
    @classmethod
    def objects(cls) -> "QuerySet":
        """Get model manager."""
        return QuerySet(cls)


class QuerySet:
    """Queryset for model queries."""
    
    def __init__(self, model_class: Type[Model]):
        self.model_class = model_class
    
    def all(self) -> List[Model]:
        """Get all objects."""
        return []
    
    def filter(self, **kwargs) -> "QuerySet":
        """Filter objects."""
        return self
    
    def get(self, **kwargs) -> Model:
        """Get single object."""
        return self.model_class(**kwargs)