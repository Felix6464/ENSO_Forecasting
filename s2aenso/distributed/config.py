import argparse

class ConfigVar:
    def _default_parse_fn(self, config, args):
        val = getattr(args, self.name, None)
        if val is not None:
            setattr(config, self.name, val)

    def __init__(self, add_argument_fn=None, parse_argument_fn=_default_parse_fn):
        if parse_argument_fn is ConfigVar._default_parse_fn:
            parse_argument_fn = self._default_parse_fn  # bind self
        self.add_argument_fn = add_argument_fn
        self.parse_argument_fn = parse_argument_fn

    def __set_name__(self, owner, name):
        self.name = name
        if not hasattr(owner, '_config_vars'):
            owner._config_vars = []
        owner._config_vars.append(self)

    def __get__(self, obj, objtype=None):
        return obj.dct[self.name]

    def __set__(self, obj, value):
        obj.dct[self.name] = value

    def __delete__(self, obj):
        del obj.dct[self.name]

    def add_argument(self, config, parser):
        if self.add_argument_fn:
            self.add_argument_fn(config, parser)

    def parse_argument(self, config, args):
        if self.parse_argument_fn:
            self.parse_argument_fn(config, args)


class ArgparseVar(ConfigVar):
    def _default_add_fn(self, config, parser):
        option = f'--{self.name.replace("_", "-")}'
        args = self.args or [option]
        kwargs = self.kwargs.copy()
        kwargs['dest'] = self.name
        parser.add_argument(*args, **kwargs)

    def __init__(self, *args, add_argument_fn=_default_add_fn, parse_argument_fn=ConfigVar._default_parse_fn, **kwargs):
        if add_argument_fn is ArgparseVar._default_add_fn:
            add_argument_fn = self._default_add_fn  # bind self

        super().__init__(add_argument_fn, parse_argument_fn)
        self.args = args
        self.kwargs = kwargs
        if 'dest' in self.kwargs:
            raise ValueError('dest cannot be specified in kwargs')


class SubConfig:
    def __init__(self, parent, root_dct, key=None):
        self.parent = parent
        self.root_dct = root_dct
        self.key = key
        self.set_defaults()

    def set_defaults(self):
        pass

    @property
    def dct(self):
        if self.key is None:
            return self.root_dct
        else:
            return self.root_dct[self.key]

    def add_arguments(self, parser):
        group = parser.add_argument_group(self.key)
        for cfg_var in self._config_vars:
            cfg_var.add_argument(self, group)

    def parse_args(self, args):
        for cfg_var in self._config_vars:
            cfg_var.parse_argument(self, args)

class BaseConfig:
    def __init__(self, dct=None):
        self._sub_configs = []
        self.dct = {}

        self._sub_configs = []
        self._setup_sub_configs()
        self.set_defaults()

        if dct:
            self.dct.update(dct)

    def __getattr__(self, name):
        if name == '_sub_configs':
            return super().__getattribute__(name)

        for cfg in self._sub_configs:
            try:
                return getattr(cfg, name)
            except AttributeError:
                pass
        raise AttributeError(f'Config has no attribute {name}')

    def __setattr__(self, name, value):
        if name != '_sub_configs':
            for cfg in self._sub_configs:
                if hasattr(cfg, name):
                    setattr(cfg, name, value)
                    return
        super().__setattr__(name, value)

    def _setup_sub_configs(self):
        raise NotImplementedError()

    def set_sub_config(self, key, cls):
        self.dct.setdefault(key, {})
        sub_cfg = cls(self, self.dct, key)
        self._sub_configs.append(sub_cfg)

    def create_parser(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        for sub_cfg in self._sub_configs:
            sub_cfg.add_arguments(parser)
        return parser

    def parse_args(self, args):
        for sub_cfg in self._sub_configs:
            sub_cfg.parse_args(args)

    def as_dictionary(self):
        return dict(self.dct)

    def set_defaults(self):
        pass

class ExperimentConfig:
    def __init__(self, config = None):
        self.cfg = {}
        if config:
            self.cfg.update(config)
                
    def __getattr__(self, name):
        return self.cfg.get(name, None)
    
    def __setattr__(self, name, value) -> None:
        if name == "cfg":
            super().__setattr__(name, value)
        else:
            self.cfg[name] = value

    def as_dictionary(self):
        return dict(self.cfg)