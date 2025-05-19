from typing import Dict, Any, Union, Optional, Type
import os


def parse_env_param(name: str, value: Any, expected_type: Type) -> Any:
    if value is None:
        return None

    try:
        if expected_type == bool:
            return value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes")
        elif expected_type == int:
            return int(value)
        elif expected_type == float:
            return float(value)
        elif expected_type == str:
            return str(value)
        elif expected_type == list:
            return [item.strip() for item in str(value).split(',') if item.strip()]
    except (ValueError, TypeError):
        pass

    raise ValueError(f"Invalid type for {name}: expected {expected_type.__name__}, got {type(value).__name__}")


def read_config_param(name: str, expected_type: Type, provided_value: Optional[Any] = None, default_value: Optional[Any] = None, required: bool = False) -> Any:
    # Check if the value was provided as an argument
    if provided_value is not None:
        return provided_value

    # Check if the value was provided as an environment variable
    env_value = os.getenv(f'GRAPHSIGNAL_{name.upper()}')
    if env_value is not None:
        parsed_env_value = parse_env_param(name, env_value, expected_type)
        if parsed_env_value is not None:
            return parsed_env_value

    if required:
        raise ValueError(f"Missing required argument: {name}")

    return default_value


def read_config_tags(provided_value: Optional[dict] = None, prefix: str = "GRAPHSIGNAL_TAG_") -> Dict[str, str]:
    # Check if the value was provided as an argument
    if provided_value is not None:
        return provided_value

    # Check if the value was provided as an environment variable
    return {key[len(prefix):].lower(): value for key, value in os.environ.items() if key.startswith(prefix)}
