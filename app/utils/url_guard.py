"""SSRF guard — reject URLs that are not public http(s) (FIX-2026-09-29-081).

Single choke point shared by ``web_extractor.fetch_page`` (the /fetch_url route)
and the Selenium credit-check navigation. Policy: only ``http``/``https`` to hosts
that resolve **exclusively** to globally-routable addresses. Loopback, private
(10/172.16/192.168), link-local (169.254, incl. the cloud metadata IP
169.254.169.254), CGNAT, multicast, reserved and unspecified addresses are all
blocked; known container/service hostnames are blocked by name as well.

Residual risk: DNS is resolved before the request, so a rebinding attacker could
still flip the record between check and connect (TOCTOU). Redirects are re-checked
per hop, and a Selenium client-side redirect is re-checked via ``current_url``.
"""
import ipaddress
import logging
import socket
from typing import Tuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Service/container names that must never be reachable (DNS may or may not resolve
# them; block by name regardless).
_DENY_HOSTNAMES = {
    'localhost', 'localhost.localdomain', 'ip6-localhost', 'ip6-loopback',
    'metadata', 'metadata.google.internal', 'metadata.goog',
    'app', 'postgres', 'redis', 'nginx', 'celery-worker', 'celery-beat',
    'host.docker.internal', 'gateway.docker.internal',
}


def _ip_blocked(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    if getattr(addr, 'ipv4_mapped', None):
        addr = addr.ipv4_mapped
    # is_global is False for loopback / private / link-local / multicast / reserved
    # / unspecified / CGNAT — i.e. exactly the set we must refuse.
    return not addr.is_global


def check_url(url: str) -> Tuple[bool, str]:
    """Return ``(ok, reason)``. ``ok`` is True only for a public http(s) URL."""
    try:
        parts = urlsplit(url)
    except Exception as e:
        return False, f'unparsable url: {e}'
    if parts.scheme not in ('http', 'https'):
        return False, f'scheme not allowed: {parts.scheme!r}'
    host = parts.hostname
    if not host:
        return False, 'missing host'
    if host.lower() in _DENY_HOSTNAMES:
        return False, f'blocked host: {host}'
    # FIX-2026-10-04-QA-030: accessing .port raises ValueError for out-of-range
    # ports (e.g. :99999). It must not escape the (ok, reason) contract.
    try:
        port = parts.port or (443 if parts.scheme == 'https' else 80)
    except ValueError as e:
        return False, f'invalid port: {e}'
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except Exception as e:
        return False, f'dns resolution failed: {e}'
    ips = {info[4][0] for info in infos}
    if not ips:
        return False, 'no addresses resolved'
    for ip in ips:
        if _ip_blocked(ip):
            return False, f'blocked ip: {ip}'
    return True, ''
