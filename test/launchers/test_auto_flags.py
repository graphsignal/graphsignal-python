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


def _change(name, value=None, action='set', change_type='arg', reason='r'):
    return {
        'type': change_type,
        'action': action,
        'name': name,
        'value': value,
        'reason': reason,
        'bottleneck_ref': None,
    }


def _ok_response(proposed_changes):
    return {
        'error': None,
        'result': {
            'bottlenecks': [],
            'proposed_changes': proposed_changes,
            'status': 'ok',
            'status_reason': None,
        },
    }


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


class RemoveFlagTest(unittest.TestCase):
    def test_removes_bare_flag(self):
        out = auto_flags._remove_flag(
            ['vllm', 'serve', '--no-cache', '--port', '8000'], '--no-cache')
        self.assertEqual(out, ['vllm', 'serve', '--port', '8000'])

    def test_removes_space_form_flag_and_value(self):
        out = auto_flags._remove_flag(
            ['vllm', 'serve', '--max-output-tokens', '10', '--port', '8000'],
            '--max-output-tokens')
        self.assertEqual(out, ['vllm', 'serve', '--port', '8000'])

    def test_removes_equals_form_flag(self):
        out = auto_flags._remove_flag(
            ['vllm', 'serve', '--max-output-tokens=10', '--port', '8000'],
            '--max-output-tokens')
        self.assertEqual(out, ['vllm', 'serve', '--port', '8000'])

    def test_noop_when_flag_absent(self):
        args = ['vllm', 'serve', '--port', '8000']
        self.assertEqual(auto_flags._remove_flag(args, '--no-cache'), args)


class MergeArgsTest(unittest.TestCase):
    def test_merges_set_changes(self):
        out = auto_flags._merge_args(
            ['vllm', 'serve', 'm'],
            [
                _change('--max-output-tokens', '123'),
                _change('--no-cache'),
            ])
        self.assertEqual(
            out, ['vllm', 'serve', 'm', '--max-output-tokens', '123', '--no-cache'])

    def test_modifies_existing_flag(self):
        out = auto_flags._merge_args(
            ['vllm', 'serve', '--max-num-seqs', '64'],
            [_change('--max-num-seqs', '48')])
        self.assertEqual(out, ['vllm', 'serve', '--max-num-seqs', '48'])

    def test_removes_flag(self):
        out = auto_flags._merge_args(
            ['vllm', 'serve', '--aefasdfasd', '--port', '8000'],
            [_change('--aefasdfasd', action='remove')])
        self.assertEqual(out, ['vllm', 'serve', '--port', '8000'])

    def test_defaults_missing_action_to_set(self):
        out = auto_flags._merge_args(
            ['vllm'],
            [{'type': 'arg', 'name': '--no-cache', 'value': None, 'reason': 'r'}])
        self.assertEqual(out, ['vllm', '--no-cache'])

    def test_skips_non_arg_changes(self):
        out = auto_flags._merge_args(
            ['vllm', 'serve'],
            [
                _change('gpu_count', '2', change_type='infra'),
                _change('--no-cache'),
            ])
        self.assertEqual(out, ['vllm', 'serve', '--no-cache'])

    def test_ignores_non_dict_and_invalid(self):
        out = auto_flags._merge_args(
            ['vllm'],
            ['not-a-dict', {}, {'type': 'arg'}, {'type': 'arg', 'name': ''}])
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
                # mem_total requires discrete VRAM; unified-memory GPUs omit it.
                mem_total_key = f'device.{i}.mem_total'
                if mem_total_key in attrs:
                    self.assertGreater(attrs[mem_total_key], 0)


class CollectSingleDeviceAttrsTest(unittest.TestCase):
    def _mock_pynvml(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetName.return_value = 'Test GPU'
        mock_pynvml.nvmlDeviceGetArchitecture.return_value = 7
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 0)
        return mock_pynvml

    def test_uses_nvml_memory_v2(self):
        attrs = {}

        def add(name, value):
            attrs[name] = value

        mock_pynvml = self._mock_pynvml()
        mock_handle = MagicMock()
        mock_mem = MagicMock()
        mock_mem.total = 128_000_000_000
        mock_pynvml.nvmlDeviceGetMemoryInfo_v2.return_value = mock_mem

        auto_flags._collect_single_device_attrs(add, mock_pynvml, mock_handle, 0)

        self.assertEqual(attrs.get('device.0.mem_total'), 128_000_000_000)
        mock_pynvml.nvmlDeviceGetMemoryInfo_v2.assert_called_once_with(mock_handle)
        mock_pynvml.nvmlDeviceGetMemoryInfo.assert_not_called()

    def test_falls_back_to_memory_info_v1(self):
        attrs = {}

        def add(name, value):
            attrs[name] = value

        mock_pynvml = self._mock_pynvml()
        mock_handle = MagicMock()
        mock_mem = MagicMock()
        mock_mem.total = 64_000_000_000
        mock_pynvml.nvmlDeviceGetMemoryInfo_v2.side_effect = Exception('Not Supported')
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        auto_flags._collect_single_device_attrs(add, mock_pynvml, mock_handle, 0)

        self.assertEqual(attrs.get('device.0.mem_total'), 64_000_000_000)
        mock_pynvml.nvmlDeviceGetMemoryInfo.assert_called_once_with(mock_handle)

    def test_omits_mem_total_when_nvml_unavailable(self):
        attrs = {}

        def add(name, value):
            attrs[name] = value

        mock_pynvml = self._mock_pynvml()
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryInfo_v2.side_effect = Exception('Not Supported')
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = Exception('Not Supported')

        auto_flags._collect_single_device_attrs(add, mock_pynvml, mock_handle, 0)

        self.assertNotIn('device.0.mem_total', attrs)


class InjectAutoFlagsTest(unittest.TestCase):
    def test_returns_args_unchanged_without_api_key(self):
        with patch.dict('os.environ', {}, clear=True):
            out = auto_flags.inject_auto_flags(['vllm', 'serve', 'm'])
        self.assertEqual(out, ['vllm', 'serve', 'm'])

    def test_applies_returned_flags(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = _ok_response([
            _change('--no-cache', reason='Disable cache.'),
            _change('--max-num-seqs', '48', reason='Lower concurrency.'),
        ])
        with patch.dict('os.environ', {'GRAPHSIGNAL_API_KEY': 'k'}, clear=True), \
             patch.object(auto_flags, '_collect_system', return_value=[]), \
             patch.object(auto_flags.requests, 'post', return_value=resp) as post_m:
            out = auto_flags.inject_auto_flags(
                ['vllm', 'serve', 'm', '--bad-flag'], workload_id='wl-abc123')

        self.assertEqual(
            out, ['vllm', 'serve', 'm', '--bad-flag', '--no-cache', '--max-num-seqs', '48'])
        called = post_m.call_args
        self.assertEqual(called.args[0], 'https://api.graphsignal.com/api/v1/auto_flags/')
        self.assertEqual(called.kwargs['headers'], {'X-API-Key': 'k'})
        body = called.kwargs['json']
        self.assertEqual(body['env']['command_line'], 'vllm serve m --bad-flag')
        self.assertEqual(body['env']['tags']['workload.id'], 'wl-abc123')

    def test_applies_remove_changes(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = _ok_response([
            _change('--bad-flag', action='remove', reason='Unknown flag.'),
            _change('--enable-chunked-prefill', reason='Smooth prefill.'),
        ])
        with patch.dict('os.environ', {'GRAPHSIGNAL_API_KEY': 'k'}, clear=True), \
             patch.object(auto_flags, '_collect_system', return_value=[]), \
             patch.object(auto_flags.requests, 'post', return_value=resp):
            out = auto_flags.inject_auto_flags(
                ['vllm', 'serve', 'm', '--bad-flag', '--port', '8000'])

        self.assertEqual(
            out, ['vllm', 'serve', 'm', '--port', '8000', '--enable-chunked-prefill'])

    def test_ignores_infra_changes(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = _ok_response([
            _change('gpu_count', '2', change_type='infra', reason='Need more GPUs.'),
            _change('--no-cache', reason='Disable cache.'),
        ])
        with patch.dict('os.environ', {'GRAPHSIGNAL_API_KEY': 'k'}, clear=True), \
             patch.object(auto_flags, '_collect_system', return_value=[]), \
             patch.object(auto_flags.requests, 'post', return_value=resp):
            out = auto_flags.inject_auto_flags(['vllm', 'serve', 'm'])

        self.assertEqual(out, ['vllm', 'serve', 'm', '--no-cache'])

    def test_uses_api_base_override(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = _ok_response([])
        with patch.dict('os.environ',
                        {'GRAPHSIGNAL_API_KEY': 'k',
                         'GRAPHSIGNAL_API_BASE': 'http://localhost:8080'}, clear=True), \
             patch.object(auto_flags, '_collect_system', return_value=[]), \
             patch.object(auto_flags.requests, 'post', return_value=resp) as post_m:
            auto_flags.inject_auto_flags(['vllm'])
        self.assertEqual(
            post_m.call_args.args[0], 'http://localhost:8080/api/v1/auto_flags/')

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
        resp.json.return_value = _ok_response([])

        def fake_post(url, json=None, headers=None, timeout=None):
            captured['body'] = json
            return resp

        with patch.dict('os.environ', {'GRAPHSIGNAL_API_KEY': 'k'}, clear=True), \
             patch.object(auto_flags, '_collect_device_attrs'), \
             patch.object(auto_flags.requests, 'post', side_effect=fake_post):
            auto_flags.inject_auto_flags(
                ['vllm'], engine_name='vllm', engine_version='1.2.3')

        system = {attr['name']: attr['value']
                  for attr in captured['body']['env']['system']}
        self.assertEqual(system.get('engine.name'), 'vllm')
        self.assertEqual(system.get('engine.version'), '1.2.3')

    def test_returns_args_unchanged_on_api_error(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {'error': 'Workload ID is required', 'result': None}
        with patch.dict('os.environ', {'GRAPHSIGNAL_API_KEY': 'k'}, clear=True), \
             patch.object(auto_flags, '_collect_system', return_value=[]), \
             patch.object(auto_flags.requests, 'post', return_value=resp):
            out = auto_flags.inject_auto_flags(['vllm', 'serve', 'm'])
        self.assertEqual(out, ['vllm', 'serve', 'm'])


if __name__ == '__main__':
    unittest.main()
