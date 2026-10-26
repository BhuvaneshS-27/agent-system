"""Why does GDELT time out from Python while PowerShell can reach it?  Prints facts only, no secrets.

  python scripts/check_gdelt.py

Your last run showed: the TCP connection to GDELT opens in 0.3 s, but Python's HTTPS request then fails after
about 5.5 s, which is the 5-second limit of the TLS handshake. This script tests that directly:

1. Python and OpenSSL versions
2. the TLS handshake with the default settings (15 s limit): how long, which TLS version
3. the TLS handshake limited to TLS 1.2 (works around some firewalls that dislike the newer ClientHello)
4. one real request with generous limits (15 s connect, 30 s read), as the agent would send it
5. whether Python is told to use a proxy (shown without any user name or password)

GDELT allows one request per 5 seconds per IP address. An answer that starts with 'Please limit requests'
means: reachable, but your IP address is being rate-limited.
"""
import socket
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests                                       # noqa: E402

HOST = "api.gdeltproject.org"
URL = f"https://{HOST}/api/v2/doc/doc"
PARAMS = {"query": "stripe", "mode": "artlist", "format": "json", "maxrecords": 1}


def handshake(label: str, max_version=None) -> None:
    ctx = ssl.create_default_context()
    if max_version is not None:
        ctx.maximum_version = max_version
    t = time.time()
    try:
        with socket.create_connection((HOST, 443), timeout=15) as sock:
            t_connect = time.time() - t
            with ctx.wrap_socket(sock, server_hostname=HOST) as tls:
                print(f"   {label}: connected in {t_connect:.2f} s, TLS handshake done at {time.time() - t:.2f} s "
                      f"({tls.version()}, {tls.cipher()[0]})")
    except Exception as e:                            # noqa: BLE001 - this is a diagnostic: show everything
        print(f"   {label}: FAILED after {time.time() - t:.1f} s ({type(e).__name__}: {str(e)[:70]})")


def main() -> int:
    print(f"1. Python {sys.version.split()[0]}, {ssl.OPENSSL_VERSION}")
    print(f"2/3. TLS handshake to {HOST}")
    try:
        infos = socket.getaddrinfo(HOST, 443, type=socket.SOCK_STREAM)
    except OSError as e:
        print(f"   DNS lookup failed: {e}")
        return 1
    print("   addresses: " + ", ".join(sorted({i[4][0] for i in infos})))
    handshake("default TLS      ")
    handshake("TLS 1.2 at most  ", ssl.TLSVersion.TLSv1_2)

    print("4. one request, generous limits (connect 15 s, read 30 s)")
    t = time.time()
    try:
        r = requests.get(URL, params=PARAMS, timeout=(15, 30), headers={"User-Agent": "Mozilla/5.0 (check_gdelt)"})
        print(f"   HTTP {r.status_code} after {time.time() - t:.1f} s   answer starts: {' '.join(r.text.split())[:90]!r}")
    except requests.ConnectTimeout:
        print(f"   CONNECT TIMEOUT after {time.time() - t:.1f} s")
    except requests.ReadTimeout:
        print(f"   READ TIMEOUT after {time.time() - t:.1f} s")
    except requests.RequestException as e:
        print(f"   {type(e).__name__} after {time.time() - t:.1f} s: {str(e)[:100]}")

    print("5. proxy settings Python sees")
    proxies = urllib.request.getproxies()
    if not proxies:
        print("   none")
    for scheme, value in proxies.items():
        p = urlsplit(value if "://" in value else f"//{value}")
        print(f"   {scheme}: {p.hostname}:{p.port}" + ("  (with login, hidden)" if p.username else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())