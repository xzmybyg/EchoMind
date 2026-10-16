from dataclasses import replace
import hashlib
import threading

import httpx
import pytest

from echomind import knowledge_model
from echomind.components import ComponentError


def test_cpu_embedder_uses_only_local_pinned_model(monkeypatch, tmp_path):
    import fastembed
    seen = []
    monkeypatch.setattr(knowledge_model, 'model_ready', lambda folder: True)
    monkeypatch.setattr(fastembed, 'TextEmbedding', lambda **kwargs: seen.append(kwargs))
    knowledge_model.LocalEmbedder(tmp_path)
    assert seen[0]['local_files_only'] is True
    assert seen[0]['providers'] == ['CPUExecutionProvider']
    assert seen[0]['specific_model_path'] == str(tmp_path)


@pytest.mark.parametrize('bad_digest', [False, True])
def test_model_download_verifies_checksum_before_publishing(monkeypatch, tmp_path, bad_digest):
    data = b'fixture model bytes'
    digest = hashlib.sha256(data).hexdigest()
    entry = {'filename':'model_optimized.onnx', 'size':len(data), 'sha256':'0'*64 if bad_digest else digest}
    monkeypatch.setattr(knowledge_model, '_data', {'revision':'fixed-revision','files':[entry]})
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=data))) as client:
        if bad_digest:
            with pytest.raises(ComponentError, match='SHA-256'):
                knowledge_model.install_model(tmp_path, client, lambda *args: None, threading.Event())
            assert not knowledge_model.model_path(tmp_path).exists()
        else:
            knowledge_model.install_model(tmp_path, client, lambda *args: None, threading.Event())
            assert knowledge_model.model_ready(knowledge_model.model_path(tmp_path))


def test_model_cancel_does_not_publish_or_contact_network(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail('cancelled download must not connect'))) as client:
        with pytest.raises(ComponentError, match='取消'):
            knowledge_model.install_model(tmp_path, client, lambda *args: None, cancel)
    assert not knowledge_model.model_path(tmp_path).exists()
