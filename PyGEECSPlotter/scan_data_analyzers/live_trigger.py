# GEECS master-control multicast, used by LiveShotWatcher as a shot clock.
#
# Each experiment's master control multicasts, on its own group:
#   ScanStart>>ControlRoom>>CAM-CR-Beampointing1 exposure>>192.168.14.2
#   Path>>\\192.168.13.6\pdna2\data\ControlRoom\Y2026\11-Nov\26_1104\scans\Scan002\xxx\Scan002_xxx_001.fileextension
#   ScanEnd>>ControlRoom>>CAM-CR-Beampointing1 exposure>>192.168.14.2
# plus 'Empty' heartbeats from other senders. 'Path>>' is a template (xxx and
# .fileextension are placeholders): it names the day folder, scan and shot,
# about 0.3-2 s before that shot's files and .tdms row exist. ScanEnd carries
# no scan number: it ends the scan its sender last named.
#
# TriggerListener receives them in a thread, keeps those for the watched
# experiment, and queues them as TriggerEvents for the watcher's loop.

import queue
import re
import socket
import struct
import threading
import time

_SHOT = re.compile(r'^(?P<top>.+)[\\/]scans[\\/]Scan(?P<scan>\d{3,})[\\/][^\\/]+[\\/]'
                   r'Scan\d{3,}_.+_(?P<shot>\d+)\.[^.\\/]+$')


class TriggerEvent:
    """One master-control message: ``kind`` is 'start', 'shot' or 'end'; for a
    shot, ``top`` (local day folder), ``scan`` and ``shot``."""

    def __init__(self, kind, t, sender, text, top=None, scan=None, shot=None):
        self.kind, self.t, self.sender, self.text = kind, t, sender, text
        self.top, self.scan, self.shot = top, scan, shot

    def __repr__(self):
        return (f'TriggerEvent({self.kind}, scan={self.scan}, shot={self.shot}, from {self.sender})'
                if self.kind == 'shot' else f'TriggerEvent({self.kind}, from {self.sender})')


def parse_group(trigger):
    """``'234.5.6.17:58456'`` or ``('234.5.6.17', 58456)`` -> ``(group, port)``."""
    if isinstance(trigger, str):
        group, port = trigger.rsplit(':', 1)
        return group, int(port)
    group, port = trigger
    return group, int(port)


class TriggerListener:
    r"""
    Receive master-control messages on ``group:port`` in a thread.

    Parameters
    ----------
    trigger : str or (str, int)
        The experiment's group, e.g. ``'234.5.6.17:58456'``.
    accept : callable
        ``accept(top)``: True for a day folder (local path) of the watched
        experiment. Other experiments' messages are ignored.
    root_map : (str, str), optional
        ``(remote_root, local_root)``: the experiment's root path as master
        control sends it and the local folder it maps to, both from the
        experiment's GEECS database, e.g.
        ``(r'\\192.168.13.6\pdna2\data\ControlRoom', r'N:\data\ControlRoom')``.
        None: the message paths are used as they are.
    wake : threading.Event, optional
        Set whenever an event is queued.

    ``drain()`` returns the queued events. ScanStart / ScanEnd (which carry no
    path) are kept only from senders whose Path messages were accepted.
    """

    def __init__(self, trigger, accept, root_map=None, wake=None):
        self.group, self.port = parse_group(trigger)
        self.accept = accept
        self.root_map = root_map
        self.wake = wake
        self.events = queue.Queue()
        self.packets = 0                 # every packet received
        self.malformed = 0               # Path messages not understood
        self.rejected = 0                # Path messages for another folder
        self.last_rejected = None        # the latest of those (as mapped to a local path)
        self.last_packet = None          # monotonic time of the last packet
        self.senders = set()             # senders of accepted Path messages
        self.error = None
        self._stop = threading.Event()
        self._sock = None
        self._thread = None

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.port))
        mreq = struct.pack('4sl', socket.inet_aton(self.group), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(0.5)            # so stop() is noticed
        self._sock = sock
        self._thread = threading.Thread(target=self._run, daemon=True, name='geecs-trigger')
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def to_local(self, path):
        """The message's path moved from the remote root to the local one
        (unchanged without a ``root_map``, or if it isn't under the remote root)."""
        if self.root_map:
            remote, local = (r.rstrip('\\/') for r in self.root_map)
            if path.lower().startswith(remote.lower()) and path[len(remote):][:1] in ('\\', '/'):
                return local + path[len(remote):]
        return path

    def parse(self, text, sender, t):
        """The TriggerEvent for one message, or None (ignored)."""
        text = text.strip().strip('\x00').strip()
        if text.startswith('Path>>'):
            m = _SHOT.match(self.to_local(text[len('Path>>'):].strip()))
            if not m:
                self.malformed += 1
                return None
            if not self.accept(m.group('top')):
                self.rejected += 1
                self.last_rejected = m.group('top')
                return None
            self.senders.add(sender)
            return TriggerEvent('shot', t, sender, text, m.group('top'), int(m.group('scan')),
                                int(m.group('shot')))
        for prefix, kind in (('ScanStart', 'start'), ('ScanEnd', 'end')):
            if text.startswith(prefix) and sender in self.senders:
                return TriggerEvent(kind, t, sender, text)
        return None

    def _run(self):
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as err:
                if not self._stop.is_set():
                    self.error = err
                return
            t = time.monotonic()
            self.packets += 1
            self.last_packet = t
            event = self.parse(data.decode('utf-8', errors='replace'), addr[0], t)
            if event is not None:
                self.events.put(event)
                if self.wake is not None:
                    self.wake.set()
