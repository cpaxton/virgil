import yaml


def str_presenter(dumper, data):
    if len(data) > 80 or "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)


def yaml_dump(data, stream=None, **kwargs):
    """Dump data to YAML format with custom string representation."""
    return yaml.dump(
        data,
        stream=stream,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        **kwargs,
    )  # type: ignore
