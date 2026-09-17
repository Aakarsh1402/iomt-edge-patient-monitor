"""A tiny MQTT 3.1.1 broker in pure Python, so the live pipeline runs on any
laptop without installing Mosquitto or Docker.

    python fusion/mini_broker.py                # listens on 0.0.0.0:1883
    python fusion/mini_broker.py --port 1884

Supports what this project's nodes need: CONNECT, PUBLISH (QoS 0/1/2 inbound,
delivered at QoS 0), SUBSCRIBE/UNSUBSCRIBE with `+` and `#` wildcards,
retained messages and PING. No auth, persistence or TLS - it is a dev tool.
The production stack is still Mosquitto (see docker-compose in the README).

    from mini_broker import Broker
    with Broker(port=0) as b:            # port 0 = pick a free port
        client.connect("127.0.0.1", b.port)
"""
import argparse
import socket
import struct
import threading

CONNECT, CONNACK, PUBLISH, PUBACK, PUBREC, PUBREL, PUBCOMP = 1, 2, 3, 4, 5, 6, 7
SUBSCRIBE, SUBACK, UNSUBSCRIBE, UNSUBACK, PINGREQ, PINGRESP, DISCONNECT = 8, 9, 10, 11, 12, 13, 14


def topic_matches(pattern, topic):
    """MQTT filter match: `+` is one level, `#` the remainder."""
    p, t = pattern.split("/"), topic.split("/")
    for i, part in enumerate(p):
        if part == "#":
            return True
        if i >= len(t) or (part != "+" and part != t[i]):
            return False
    return len(p) == len(t)


def encode_remaining_length(n):
    out = bytearray()
    while True:
        byte, n = n % 128, n // 128
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def publish_packet(topic, payload, retain=False):
    tb = topic.encode()
    body = struct.pack("!H", len(tb)) + tb + payload
    return bytes([(PUBLISH << 4) | int(retain)]) + encode_remaining_length(len(body)) + body


def _read_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed")
        buf += chunk
    return buf


def read_packet(sock):
    """-> (type, flags, body) or raises ConnectionError."""
    first = _read_exact(sock, 1)[0]
    length, mult = 0, 1
    while True:
        b = _read_exact(sock, 1)[0]
        length += (b & 0x7F) * mult
        mult *= 128
        if not b & 0x80:
            break
    return first >> 4, first & 0x0F, _read_exact(sock, length)


def _string(body, pos):
    n = struct.unpack_from("!H", body, pos)[0]
    return body[pos + 2:pos + 2 + n].decode(errors="replace"), pos + 2 + n


class Broker:
    def __init__(self, host="0.0.0.0", port=1883, verbose=False):
        self.host, self.port, self.verbose = host, port, verbose
        self._lock = threading.Lock()
        self._subs = {}                 # socket -> set(filters)
        self._retained = {}             # topic -> payload
        self._server = None
        self._thread = None
        self.messages = 0

    # -- lifecycle -------------------------------------------------------
    def start(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self.port = self._server.getsockname()[1]
        self._server.listen(16)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._server:
            try:
                self._server.close()
            finally:
                self._server = None
        with self._lock:
            for sock in list(self._subs):
                sock.close()
            self._subs.clear()

    __enter__ = start

    def __exit__(self, *exc):
        self.stop()

    # -- internals -------------------------------------------------------
    def _accept_loop(self):
        while self._server:
            try:
                sock, _addr = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(sock,), daemon=True).start()

    def _send(self, sock, data):
        try:
            sock.sendall(data)
        except OSError:
            self._drop(sock)

    def _drop(self, sock):
        with self._lock:
            self._subs.pop(sock, None)
        try:
            sock.close()
        except OSError:
            pass

    def _deliver(self, topic, payload, retain):
        self.messages += 1
        with self._lock:
            if retain:
                if payload:
                    self._retained[topic] = payload
                else:
                    self._retained.pop(topic, None)
            targets = [s for s, filters in self._subs.items() if any(topic_matches(f, topic) for f in filters)]
        packet = publish_packet(topic, payload)
        for s in targets:
            self._send(s, packet)

    def _serve(self, sock):
        try:
            ptype, _flags, body = read_packet(sock)
            if ptype != CONNECT:
                return
            _name, pos = _string(body, 0)
            client_id, _ = _string(body, pos + 4)      # level, flags, keepalive = 4 bytes
            if self.verbose:
                print(f"[broker] {client_id or '<anon>'} connected")
            self._send(sock, bytes([CONNACK << 4, 2, 0, 0]))
            with self._lock:
                self._subs[sock] = set()
            while True:
                ptype, flags, body = read_packet(sock)
                if ptype == PUBLISH:
                    qos = (flags >> 1) & 3
                    topic, pos = _string(body, 0)
                    if qos:
                        pid = body[pos:pos + 2]
                        pos += 2
                        self._send(sock, bytes([(PUBACK if qos == 1 else PUBREC) << 4, 2]) + pid)
                    self._deliver(topic, body[pos:], bool(flags & 1))
                elif ptype == PUBREL:
                    self._send(sock, bytes([PUBCOMP << 4, 2]) + body[:2])
                elif ptype == SUBSCRIBE:
                    pid, pos, granted = body[:2], 2, bytearray()
                    while pos < len(body):
                        filt, pos = _string(body, pos)
                        pos += 1                                    # requested QoS
                        granted.append(0)
                        with self._lock:
                            self._subs[sock].add(filt)
                            retained = [(t, p) for t, p in self._retained.items() if topic_matches(filt, t)]
                        for t, p in retained:
                            self._send(sock, publish_packet(t, p, retain=True))
                    self._send(sock, bytes([SUBACK << 4]) + encode_remaining_length(2 + len(granted)) + pid + granted)
                elif ptype == UNSUBSCRIBE:
                    pid, pos = body[:2], 2
                    while pos < len(body):
                        filt, pos = _string(body, pos)
                        with self._lock:
                            self._subs[sock].discard(filt)
                    self._send(sock, bytes([UNSUBACK << 4, 2]) + pid)
                elif ptype == PINGREQ:
                    self._send(sock, bytes([PINGRESP << 4, 0]))
                elif ptype == DISCONNECT:
                    return
        except (ConnectionError, OSError, struct.error, IndexError):
            pass
        finally:
            self._drop(sock)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    broker = Broker(args.host, args.port, verbose=not args.quiet).start()
    print(f"mini MQTT broker listening on {args.host}:{broker.port} (Ctrl-C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        broker.stop()


if __name__ == "__main__":
    main()
