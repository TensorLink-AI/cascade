"""Fail a build if Python's default or installed certifi stores retain affected roots."""
from __future__ import annotations

import hashlib
import importlib
import json
import ssl
import sys

AFFECTED_ROOTS = {
    '7DD78D5F4F13459A83DFF9ABBBA62EDBAF6F2D102BF257FD712F4D9F2746ED8D',
    'F2EDAAFF3D9B701FB4D14C9F285F8C0BAD90FC3BCC1FA07D47F4718EDABA34A5',
    'D7A7A0FB5D7E2731D771E9484EBCDEF71D5F0C3E0A2948782BC83EE0EA699EF4',
    '86A1ECBA089C4A8D3BBE2734C612BA341D813E043CF9E8A862CD5C57A36BBE6B',
}


def main():
    contexts = {'system': ssl.create_default_context()}
    for module_name in ['certifi', 'pip._vendor.certifi']:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        contexts[module_name] = ssl.create_default_context(cafile=module.where())
    results = []
    for name, context in contexts.items():
        found = sorted(AFFECTED_ROOTS & {
            hashlib.sha256(cert).hexdigest().upper()
            for cert in context.get_ca_certs(binary_form=True)
        })
        results.append({'store': name, 'affected_roots': found})
    print(json.dumps({'python': sys.version.split()[0], 'stores': results}))
    return int(any(row['affected_roots'] for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
