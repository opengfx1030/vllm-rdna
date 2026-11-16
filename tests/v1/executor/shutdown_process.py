# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spawn target without importing torch or executor modules in the child."""

import signal


def shutdown_process(connection, ignore_term):
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    connection.send("ready")
    connection.recv()
    connection.close()
