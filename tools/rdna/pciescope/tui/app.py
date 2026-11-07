# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pciescope dashboard (textual).

Layout (works at 80x24, scales up): header, tab line, one view, key line.
Views are rendered by the pure functions of ``render``; this module only wires
data sources, keys and timers.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import ContentSwitcher, Static

from pciescope.model import Record
from pciescope.tui import render
from pciescope.tui.measure import MeasureRunner
from pciescope.tui.render import HeaderInfo, MatrixSel
from pciescope.tui.sources import LiveSource, ReplaySource, Source
from pciescope.tui.state import ViewState

HELP = """\
 pciescope TUI keys

  1-5        topology / live / matrix / errors / measure
  tab        next view            shift+tab  previous view
  space      pause (live: freeze the display, recording continues)
  e          export the session as a Perfetto / chrome trace
  r          live: stop / start recording
  ← →        replay: seek -/+ 10 s (shift: 60 s)    home/end  jump
  + -        replay: speed x2 / x0.5
  d          topology: show the bridge chain of every GPU
  m          matrix: bandwidth / efficiency / latency
  [ ]        matrix: smaller / larger message size
  v          matrix: direction (push / pull / sdma / h2d / d2h)
  ↑ ↓        measure: choose suite     g  choose GPU set
  s          measure: start (refused unless the lease is held)
  x          measure: cancel          c  errors: clear delta log
  ?          this help               q  quit

 Passive collectors only read sysfs; probes run only on leased GPUs.
 Press any key to close.
"""


class HelpScreen(ModalScreen[None]):
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > Static { width: auto; max-width: 100%; height: auto;
                          background: $panel; padding: 0 1; border: round $accent; }
    """

    def compose(self) -> ComposeResult:
        yield Static(Text(HELP))

    def on_key(self, event) -> None:
        event.stop()
        self.dismiss(None)


class PcieScopeApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #hdr, #tabs, #keys { height: 1; }
    #body { height: 1fr; }
    .view { height: 1fr; }
    .view { scrollbar-size-vertical: 1; }
    .view > Static { width: 100%; }
    """
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("1", "view('topology')", show=False),
        Binding("2", "view('live')", show=False),
        Binding("3", "view('matrix')", show=False),
        Binding("4", "view('errors')", show=False),
        Binding("5", "view('measure')", show=False),
        Binding("tab", "next_view(1)", show=False, priority=True),
        Binding("shift+tab", "next_view(-1)", show=False, priority=True),
        Binding("pageup", "scroll(-1)", show=False),
        Binding("pagedown", "scroll(1)", show=False),
        Binding("space", "pause", show=False),
        Binding("e", "export", show=False),
        Binding("r", "toggle_record", show=False),
        Binding("left", "left", show=False),
        Binding("right", "right", show=False),
        Binding("shift+left", "seek(-60)", show=False),
        Binding("shift+right", "seek(60)", show=False),
        Binding("home", "seek_to(0)", show=False),
        Binding("end", "seek_end", show=False),
        Binding("plus,equals_sign", "speed(2)", show=False),
        Binding("minus", "speed(0.5)", show=False),
        Binding("d", "detail", show=False),
        Binding("m", "matrix_metric", show=False),
        Binding("left_square_bracket", "matrix_size(-1)", show=False),
        Binding("right_square_bracket", "matrix_size(1)", show=False),
        Binding("v", "matrix_direction", show=False),
        Binding("up", "up", show=False),
        Binding("down", "down", show=False),
        Binding("g", "gpus", show=False),
        Binding("s", "start", show=False),
        Binding("x", "cancel", show=False),
        Binding("c", "clear_errors", show=False),
        Binding("question_mark", "help", show=False),
    ]

    def __init__(
        self,
        source: Source,
        *,
        runner: MeasureRunner | None = None,
        export: Callable[[], str] | None = None,
        tick: float = 0.25,
        view: str = "topology",
    ) -> None:
        super().__init__()
        self.source = source
        self.runner = runner
        self._export = export
        self.tick = tick
        self.view = view
        self.state = ViewState()
        self.detail = False
        self.sel = MatrixSel()
        self.note = ""
        self._note_ticks = 0
        self._gen = getattr(source, "generation", 0)
        self._rebuild()

    # -- layout --------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static(id="hdr", markup=False)
        yield Static(id="tabs", markup=False)
        with ContentSwitcher(initial=f"v-{self.view}", id="body"):
            for v in render.VIEWS:
                with VerticalScroll(id=f"v-{v}", classes="view", can_focus=False):
                    yield Static(id=f"s-{v}", markup=False)
        yield Static(id="keys", markup=False)

    def on_mount(self) -> None:
        self.source.start()
        self.set_interval(self.tick, self._on_tick)
        self._refresh()

    def on_unmount(self) -> None:
        self.source.close()

    # -- data ----------------------------------------------------------------
    def _rebuild(self) -> None:
        """Rebuild the view state (start, or replay rewound)."""
        self.state = ViewState()
        self.state.collectors = list(self.source.collectors)
        self.state.unavailable = dict(self.source.unavailable)
        self.state.interval_ns = int(getattr(self.source, "interval", 1.0) * 1e9)
        self.state.ingest(self.source.history())
        self._fill_display()

    def _fill_display(self) -> None:
        ts = self.source.display_ns()
        if ts is not None:
            self.state.fill_to(ts)

    def _on_tick(self) -> None:
        recs: list[Record] = self.source.poll()
        gen = getattr(self.source, "generation", 0)
        if gen != self._gen:
            self._gen = gen
            self._rebuild()
        else:
            self.state.ingest(recs)
            self._fill_display()
        if self.runner is not None:
            self.runner.drain()
        if self._note_ticks:
            self._note_ticks -= 1
            if not self._note_ticks:
                self.note = ""
        self._refresh()

    # -- rendering -----------------------------------------------------------
    def header_info(self) -> HeaderInfo:
        s = self.source
        topo = s.topo
        if isinstance(s, ReplaySource):
            return HeaderInfo(
                "replay",
                topo.host,
                self.view,
                s.display_ns(),
                False,
                os.path.basename(s.path.rstrip("/")),
                s.paused,
                s.speed,
                s.pos_s,
                s.duration_s,
                self.state.n_samples,
                self.note,
            )
        assert isinstance(s, LiveSource)
        return HeaderInfo(
            "live",
            topo.host,
            self.view,
            s.display_ns(),
            s.recording,
            os.path.basename(s.session_path.rstrip("/")),
            s.paused,
            1.0,
            None,
            None,
            self.state.n_samples,
            self.note,
        )

    def body_text(self, view: str, width: int, height: int) -> Text:
        topo = self.source.topo
        if view == "topology":
            return render.render_topology(topo, self.state, width, self.detail)
        if view == "live":
            return render.render_live(topo, self.state, width)
        if view == "matrix":
            t = render.render_matrix(topo, self.state, width, self.sel)
            t.append("\n")
            t.append(render.render_other_measurements(self.state, width))
            return t
        if view == "errors":
            return render.render_errors(topo, self.state, width)
        if self.runner is None:
            t = Text(
                "Measure is disabled while replaying a recorded session.\n",
                style="yellow",
            )
            t.append(
                "Start the TUI without --replay on the box to run probes.\n",
                style="dim",
            )
            return t
        return render.render_measure(
            topo, self.runner.model, width, height, self.runner.elapsed()
        )

    def _refresh(self) -> None:
        w, h = self.size.width, self.size.height
        if w <= 0:
            return
        self.query_one("#hdr", Static).update(
            render.render_header(self.header_info(), w)
        )
        self.query_one("#tabs", Static).update(render.render_tabs(w, self.view))
        self.query_one("#keys", Static).update(
            render.render_keys(w, self.view, self.source.mode)
        )
        pane = self._pane()
        bw = pane.scrollable_content_region.width or w
        body = self.body_text(self.view, bw, h - 3)
        body.rstrip()
        self.query_one(f"#s-{self.view}", Static).update(body)

    def screen_text(self) -> str:
        """Plain text of the whole screen as a terminal would show it."""
        return screen_text(self)

    # -- actions ---------------------------------------------------------------
    def action_view(self, view: str) -> None:
        self.view = view
        self.query_one("#body", ContentSwitcher).current = f"v-{view}"
        self._refresh()

    def action_next_view(self, step: int) -> None:
        i = render.VIEWS.index(self.view)
        self.action_view(render.VIEWS[(i + step) % len(render.VIEWS)])

    def action_pause(self) -> None:
        self.source.toggle_pause()
        self._refresh()

    def _say(self, msg: str, ticks: int = 20) -> None:
        self.note, self._note_ticks = msg, ticks

    def action_export(self) -> None:
        if self._export is None:
            self._say("export unavailable")
        else:
            try:
                self._say("exported " + self._export(), 60)
            except Exception as exc:
                self._say(f"export failed: {exc}", 60)
        self._refresh()

    def action_toggle_record(self) -> None:
        if isinstance(self.source, LiveSource):
            self.source.toggle_record()
            self._say("recording " + ("on" if self.source.recording else "off"))
            self._refresh()

    def _replay(self) -> ReplaySource | None:
        return self.source if isinstance(self.source, ReplaySource) else None

    def action_seek(self, delta: float) -> None:
        rp = self._replay()
        if rp is not None:
            rp.seek(delta)
            self._on_tick()

    def action_seek_to(self, pos: float) -> None:
        rp = self._replay()
        if rp is not None:
            rp.seek_to(pos)
            self._on_tick()

    def action_seek_end(self) -> None:
        rp = self._replay()
        if rp is not None:
            rp.seek_to(rp.duration_s)
            self._on_tick()

    def action_left(self) -> None:
        self.action_seek(-10)

    def action_right(self) -> None:
        self.action_seek(10)

    def action_speed(self, factor: float) -> None:
        rp = self._replay()
        if rp is not None:
            rp.set_speed(factor)
            self._refresh()

    def action_detail(self) -> None:
        self.detail = not self.detail
        self._refresh()

    def action_matrix_metric(self) -> None:
        self.sel.cycle_view()
        self._refresh()

    def action_matrix_size(self, step: int) -> None:
        self.sel.cycle_size(step)
        self._refresh()

    def action_matrix_direction(self) -> None:
        dirs, _ = render.matrix_choices(self.state, self.sel.view)
        self.sel.cycle_direction(dirs)
        self._refresh()

    def _pane(self) -> VerticalScroll:
        return self.query_one(f"#v-{self.view}", VerticalScroll)

    def action_scroll(self, pages: int) -> None:
        self._pane().scroll_page_down() if pages > 0 else self._pane().scroll_page_up()

    def action_up(self) -> None:
        if self.runner is not None and self.view == "measure":
            self.runner.move(-1)
            self._refresh()
        else:
            self._pane().scroll_up()

    def action_down(self) -> None:
        if self.runner is not None and self.view == "measure":
            self.runner.move(1)
            self._refresh()
        else:
            self._pane().scroll_down()

    def action_gpus(self) -> None:
        if self.runner is not None and self.view == "measure":
            self.runner.cycle_gpus()
            self._refresh()

    def action_start(self) -> None:
        if self.runner is None or self.view != "measure":
            return
        err = self.runner.start()
        if err:
            self.runner.model.log.append(err)
            self._say(err, 40)
        self._refresh()

    def action_cancel(self) -> None:
        if self.runner is not None and self.view == "measure":
            self.runner.cancel()
            self._refresh()

    def action_clear_errors(self) -> None:
        if self.view == "errors":
            self.state.errors.clear()
            self._refresh()

    def action_help(self) -> None:
        self.push_screen(HelpScreen())


def screen_text(app: App) -> str:
    """Plain text of the current screen (private compositor API, test/ssh aid)."""
    strips = app.screen._compositor.render_strips()
    return "\n".join(s.text.rstrip() for s in strips)


def run(
    replay: str | None = None,
    interval: float = 1.0,
    collectors: list[str] | None = None,
    session_dir: str | None = None,
    sysfs_root: str | None = None,
    *,
    owner: str = "",
    record: bool = True,
    dry_run: bool = False,
    speed: float = 1.0,
) -> int:
    """Entry point of ``pciescope tui``; returns the process exit code."""
    from pciescope.tui import session

    try:
        setup = session.open_session(
            replay=replay,
            interval=interval,
            collectors=collectors,
            session_dir=session_dir,
            sysfs_root=sysfs_root,
            owner=owner,
            record=record,
            dry_run=dry_run,
            speed=speed,
        )
    except session.SessionError as exc:
        print(f"pciescope tui: {exc}", file=sys.stderr)
        return 2
    app = PcieScopeApp(
        setup.source,
        runner=setup.runner,
        export=setup.export,
    )
    app.run()
    return 0
