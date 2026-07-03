import unittest
from unittest.mock import MagicMock, patch

import pytest

from graphsignal.launchers import auto_flags


def has_nvidia_gpu() -> bool:
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return device_count > 0
    except Exception:
        return False


class ApplyFlagTest(unittest.TestCase):
    def test_appends_bare_flag_when_absent(self):
        out = auto_flags._apply_flag(['vllm', 'serve'], '--no-cache', ['--no-cache'])
        self.assertEqual(out, ['vllm', 'serve', '--no-cache'])

    def test_appends_flag_with_value_when_absent(self):
        out = auto_flags._apply_flag(
            ['vllm', 'serve'], '--max-output-tokens', ['--max-output-tokens', '123'])
        self.assertEqual(out, ['vllm', 'serve', '--max-output-tokens', '123'])

    def test_modifies_space_form_value(self):
        out = auto_flags._apply_flag(
            ['vllm', 'serve', '--max-output-tokens', '10'],
            '--max-output-tokens', ['--max-output-tokens', '123'])
        self.assertEqual(out, ['vllm', 'serve', '--max-output-tokens', '123'])

    def test_modifies_equals_form_value(self):
        out = auto_flags._apply_flag(
            ['vllm', 'serve', '--max-output-tokens=10'],
            '--max-output-tokens', ['--max-output-tokens', '123'])
        self.assertEqual(out, ['vllm', 'serve', '--max-output-tokens', '123'])


class MergeOptionsTest(unittest.TestCase):
    def test_merges_multiple_options(self):
        out = auto_flags._merge_options(
            ['vllm', 'serve', 'm'],
            [{'args': ['--max-output-tokens 123']}, {'args': ['--no-cache']}])
        self.assertEqual(
            out, ['vllm', 'serve', 'm', '--max-output-tokens', '123', '--no-cache'])

    def test_ignores_non_dict_and_empty(self):
        out = auto_flags._merge_options(
            ['vllm'], ['not-a-dict', {}, {'args': []}, {'args': ['']}])
        self.assertEqual(out, ['vllm'])


class CollectSystemTest(unittest.TestCase):
    @pytest.mark.cuda
    def test_collect_system(self):
        system = auto_flags._collect_system(
            engine_name='vllm', engine_version='1.2.3')
        attrs = {attr['name']: attr['value'] for attr in system}

        # Attribute list entries always carry name/value.
        for attr in system:
            self.assertIn('name', attr)
            self.assertIn('value', attr)

        # Platform + engine attributes are available on any machine.
        self.assertIn('platform.name', attrs)
        self.assertIn('platform.version', attrs)
        self.assertIn('platform.machine', attrs)
        self.assertEqual(attrs.get('engine.name'), 'vllm')
        self.assertEqual(attrs.get('engine.version'), '1.2.3')

        if has_nvidia_gpu():
            self.assertIn('device.count', attrs)
            device_count = attrs['device.count']
            self.assertGreater(device_count, 0)
            # Every visible device contributes its indexed attributes.
            for i in range(device_count):
                self.assertIn(f'device.{i}.device_name', attrs)
                self.assertIn(f'device.{i}.mem_total', attrs)


class InjectAutoFlagsTest(unittest.TestCase):
    def test_returns_args_unchanged_without_api_key(self):
        with patch.dict('os.environ', {}, clear=True):
            out = auto_flags.inject_auto_flags(['vllm', 'serve', 'm'])
        self.assertEqual(out, ['vllm', 'serve', 'm'])

    def test_applies_returned_flags(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = [{'args': ['--no-cache']}]
        with patch.dict('os.environ', {'GRAPHSIGNAL_API_KEY': 'k'}, clear=True), \
             patch.object(auto_flags, '_collect_system', return_value=[]), \
             patch.object(auto_flags.requests, 'post', return_value=resp) as post_m:
            out = auto_flags.inject_auto_flags(['vllm', 'serve', 'm'])

        self.assertEqual(out, ['vllm', 'serve', 'm', '--no-cache'])
        called = post_m.call_args
        self.assertEqual(called.args[0], 'https://api.graphsignal.com/api/v1/auto_flags')
        self.assertEqual(called.kwargs['headers'], {'X-API-Key': 'k'})
        body = called.kwargs['json']
        self.assertEqual(body['command_line'], 'vllm serve m')
        self.assertIn('host.name', body['tags'])

    def test_uses_api_base_override(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = []
        with patch.dict('os.environ',
                        {'GRAPHSIGNAL_API_KEY': 'k',
                         'GRAPHSIGNAL_API_BASE': 'http://localhost:8080'}, clear=True), \
             patch.object(auto_flags, '_collect_system', return_value=[]), \
             patch.object(auto_flags.requests, 'post', return_value=resp) as post_m:
            auto_flags.inject_auto_flags(['vllm'])
        self.assertEqual(
            post_m.call_args.args[0], 'http://localhost:8080/api/v1/auto_flags')

    def test_returns_args_unchanged_on_network_error(self):
        with patch.dict('os.environ', {'GRAPHSIGNAL_API_KEY': 'k'}, clear=True), \
             patch.object(auto_flags, '_collect_system', return_value=[]), \
             patch.object(auto_flags.requests, 'post',
                          side_effect=Exception('boom')):
            out = auto_flags.inject_auto_flags(['vllm', 'serve', 'm'])
        self.assertEqual(out, ['vllm', 'serve', 'm'])

    def test_includes_engine_attrs_when_provided(self):
        captured = {}
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = []

        def fake_post(url, json=None, headers=None, timeout=None):
            captured['body'] = json
            return resp

        with patch.dict('os.environ', {'GRAPHSIGNAL_API_KEY': 'k'}, clear=True), \
             patch.object(auto_flags, '_collect_device_attrs'), \
             patch.object(auto_flags.requests, 'post', side_effect=fake_post):
            auto_flags.inject_auto_flags(
                ['vllm'], engine_name='vllm', engine_version='1.2.3')

        system = {attr['name']: attr['value'] for attr in captured['body']['system']}
        self.assertEqual(system.get('engine.name'), 'vllm')
        self.assertEqual(system.get('engine.version'), '1.2.3')


if __name__ == '__main__':
    unittest.main()
