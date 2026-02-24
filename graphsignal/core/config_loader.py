import time
import logging
from typing import Dict, Optional, List, Callable

import requests
import graphsignal

logger = logging.getLogger('graphsignal')


class ConfigLoader:
    UPDATE_INTERVAL_SEC = 20

    def __init__(self):
        self._update_funcs = []
        self._options: Dict[str, str] = {}

    def setup(self):
        pass

    def shutdown(self):
        pass

    def clear(self):
        self._update_funcs.clear()
        self._options.clear()

    def on_update(self, update_func: Callable[[List[str]], None]):
        self._update_funcs.append(update_func)

    def emit_update(self, changed_options: List[str]):
        for update_func in self._update_funcs:
            try:
                update_func(changed_options)
            except Exception as exc:
                logger.error('Error calling update function: %s', exc, exc_info=True)

    def get_str_option(self, name: str) -> Optional[str]:
        return self._options.get(name)

    def get_int_option(self, name: str) -> Optional[int]:
        value_str = self._options.get(name)
        if value_str is None:
            return None
        try:
            return int(value_str) if value_str else None
        except (ValueError, TypeError):
            return None

    def get_float_option(self, name: str) -> Optional[float]:
        value_str = self._options.get(name)
        if value_str is None:
            return None
        try:
            return float(value_str) if value_str else None
        except (ValueError, TypeError):
            return None

    def _format_tags(self, tags: Dict[str, str]) -> str:
        if not tags:
            return None
        tag_pairs = [f"{key}:{value}" for key, value in tags.items()]
        return ";".join(tag_pairs)

    def update_config(self):
        try:
            tags_str = self._format_tags(graphsignal._ticker.tags)
            
            url = f"{graphsignal._ticker.api_url}/api/v1/sdk_config/"
            headers = {
                'X-API-Key': graphsignal._ticker.api_key
            }
            params = {}
            if tags_str:
                params['tags'] = tags_str

            resp = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=(5, 10)
            )
            resp.raise_for_status()
            config_data = resp.json()
            
            # Parse options list
            options = config_data.get('options', [])
            new_options = {opt.get('name'): opt.get('value') for opt in options}
            
            # Find changed options
            changed_options = []
            all_option_names = set(self._options.keys()) | set(new_options.keys())
            for opt_name in all_option_names:
                old_value = self._options.get(opt_name)
                new_value = new_options.get(opt_name)
                if old_value != new_value:
                    changed_options.append(opt_name)
            
            # Update options if any changed
            if changed_options:
                self._options = new_options
                try:
                    self.emit_update(changed_options)
                except Exception as exc:
                    logger.error('Error emitting update for changed options: %s', exc, exc_info=True)

            logger.debug('Fetched SDK config: %s', new_options)
        except Exception as exc:
            logger.error('Error fetching sampling config: %s', exc, exc_info=True)

