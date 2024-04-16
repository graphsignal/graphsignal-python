import uuid
import hashlib


def sanitize_str(val, max_len=250):
    if not isinstance(val, str):
        return str(val)[:max_len]
    else:
        return val[:max_len]


def sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


def uuid_sha1(size=-1):
    return sha1(str(uuid.uuid4()), size)
