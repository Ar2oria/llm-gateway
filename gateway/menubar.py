"""macOS menu bar companion for the llm-gateway.

Shows a status-bar icon while the gateway runs: left-click opens the admin
page, right-click pops up a menu (open page / restart service / stop
service). The restart entry detaches a `gateway.sh restart` run that
replaces both the gateway process and this menubar process. The app exits
by itself when the gateway process dies.

macOS-only. Install the extra first:  uv sync --extra menubar
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

PID_FILE = Path("/tmp/llm-gateway.pid")


def admin_url_for_config(config_path: str | Path) -> str:
    """Admin page URL for a config file (http://127.0.0.1:<port>/admin)."""
    try:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        port = raw.get("server", {}).get("port", 8080)
    except (OSError, ValueError):
        port = 8080
    return f"http://127.0.0.1:{port}/admin"


def gateway_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def gateway_alive() -> bool:
    pid = gateway_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_gateway() -> None:
    pid = gateway_pid()
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def restart_command(config_path: str | Path) -> list[str]:
    """gateway.sh restart argv for restarting the whole service (gateway +
    menubar) with the given config.

    The config path is absolutized: gateway.sh resolves relative configs
    against its caller's cwd, which is not the menubar's cwd.
    """
    repo_root = Path(__file__).resolve().parent.parent
    config = Path(config_path).expanduser().absolute()
    return ["bash", str(repo_root / "gateway.sh"), "restart", str(config)]


def restart_gateway(config_path: str | Path) -> None:
    """Trigger a full service restart via gateway.sh in a detached session.

    The script stops the gateway and pkills every menubar (this one
    included), then starts a fresh gateway + menubar. Detaching
    (start_new_session) keeps the script running after this process dies.
    """
    subprocess.Popen(
        restart_command(config_path),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="llm-gateway menu bar companion (macOS)")
    parser.add_argument("-c", "--config", default="config.json", help="gateway config file")
    args = parser.parse_args()
    admin_url = admin_url_for_config(args.config)

    try:
        from AppKit import (
            NSApplication,
            NSApplicationActivationPolicyAccessory,
            NSEventMaskLeftMouseUp,
            NSEventMaskRightMouseUp,
            NSEventTypeRightMouseUp,
            NSImage,
            NSMenu,
            NSMenuItem,
            NSObject,
            NSStatusBar,
            NSVariableStatusItemLength,
        )
        from Foundation import NSTimer
    except ImportError:
        print(
            "menu bar companion needs the 'menubar' extra: uv sync --extra menubar",
            file=sys.stderr,
        )
        raise SystemExit(1)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
    button = status_item.button()
    icon = NSImage.imageWithSystemSymbolName_accessibilityDescription_("bolt.fill", None)
    if icon is not None:
        icon.setTemplate_(True)
        button.setImage_(icon)
    else:
        button.setTitle_("⚡")

    menu = NSMenu.alloc().init()

    class Delegate(NSObject):
        def buttonAction_(self, sender):
            event = NSApplication.sharedApplication().currentEvent()
            if event is not None and event.type() == NSEventTypeRightMouseUp:
                status_item.popUpStatusItemMenu_(menu)
            else:
                subprocess.Popen(["open", admin_url])

        def openPage_(self, sender):
            subprocess.Popen(["open", admin_url])

        def restartService_(self, sender):
            stop_gateway()
            restart_gateway(args.config)
            app.terminate_(sender)

        def stopService_(self, sender):
            stop_gateway()
            app.terminate_(sender)

        def checkGateway_(self, sender):
            if not gateway_alive():
                app.terminate_(sender)

    delegate = Delegate.alloc().init()
    button.setTarget_(delegate)
    button.setAction_("buttonAction:")
    button.sendActionOn_(NSEventMaskLeftMouseUp | NSEventMaskRightMouseUp)

    open_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("打开管理页面", "openPage:", "")
    open_item.setTarget_(delegate)
    menu.addItem_(open_item)
    # No self-terminate on restart: the script's stop phase pkills every
    # menubar; if the script fails, the menu stays alive and is retryable.
    restart_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "重启服务", "restartService:", "")
    restart_item.setTarget_(delegate)
    menu.addItem_(restart_item)
    stop_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("关闭服务", "stopService:", "")
    stop_item.setTarget_(delegate)
    menu.addItem_(stop_item)

    # Quit when the gateway process disappears (stopped externally).
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        2.0, delegate, "checkGateway:", None, True
    )

    app.run()


if __name__ == "__main__":
    main()
