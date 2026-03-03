"""Tests for the interactive visualization module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.text import Text

from graphix.command import E, M, N, X, Z
from graphix.measurements import Measurement
from graphix.pattern import Pattern
from graphix.visualization import GraphVisualizer
from graphix.visualization_interactive import InteractiveGraphVisualizer


class TestInteractiveGraphVisualizer:
    @pytest.fixture
    def pattern(self) -> Pattern:
        """Fixture to provide a standard pattern for testing."""
        pattern = Pattern(input_nodes=[0, 1])
        pattern.add(N(node=0))
        pattern.add(N(node=1))
        pattern.add(N(node=2))
        pattern.add(E(nodes=(0, 1)))
        pattern.add(E(nodes=(1, 2)))
        pattern.add(M(node=0, measurement=Measurement.XY(0.5), s_domain={1}, t_domain={2}))
        pattern.add(M(node=1, measurement=Measurement.XY(0.0), s_domain={2}, t_domain=set()))
        pattern.add(X(node=2, domain={0}))
        pattern.add(Z(node=2, domain={1}))
        return pattern

    def test_init_and_layout(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test initialization, layout scaling, OpenGraph inference, and command_window_size parity."""
        mock_og_class = mocker.patch("graphix.visualization_interactive.OpenGraph")
        mock_og_instance = mock_og_class.return_value
        mock_og_instance.infer_pauli_measurements.return_value = mock_og_instance
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        expected_pos = {0: (10, 10), 1: (20, 20), 2: (30, 30)}
        mock_vis_obj.get_layout.return_value = (expected_pos, {}, {})

        mock_vis_obj.determine_figsize.return_value = (14.0, 7.0)
        viz = InteractiveGraphVisualizer(pattern)

        assert viz.total_steps == len(pattern)
        assert viz.enable_simulation
        assert len(viz.node_positions) == 3
        mock_vis_obj.get_layout.assert_called_once()
        mock_og_instance.infer_pauli_measurements.assert_called_once()
        for node, (ex, ey) in expected_pos.items():
            ax, ay = viz.node_positions[node]
            assert ax == pytest.approx(ex * viz.node_distance[0])
            assert ay == pytest.approx(ey * viz.node_distance[1])

        # Narrow figure: max(1, min(7, int(4.0 / 1.8))) = 2 → even → decremented to 1.
        mock_vis_obj.determine_figsize.return_value = (4.0, 7.0)
        viz_narrow = InteractiveGraphVisualizer(pattern)
        assert viz_narrow.command_window_size % 2 == 1

    def test_graph_state_with_simulation(self, mocker: MagicMock) -> None:
        """Test simulation path: topology, adaptive Clifford signals, Z-correction dict init, and label format."""
        # Node 1 uses s_domain={0} and t_domain={0}; with result[0]=1 both signals are odd,
        # exercising Clifford.X (s_signal) and Clifford.Z (t_signal) branches.
        # Z(node=3) is the first correction on node 3, exercising the dict-initialisation branch.
        sim_pattern = Pattern(input_nodes=[0, 1])
        sim_pattern.add(N(node=2))
        sim_pattern.add(N(node=3))
        sim_pattern.add(E(nodes=(0, 2)))
        sim_pattern.add(E(nodes=(1, 3)))
        sim_pattern.add(M(node=0, measurement=Measurement.XY(0.0)))
        sim_pattern.add(M(node=1, measurement=Measurement.XY(0.0), s_domain={0}, t_domain={0}))
        sim_pattern.add(X(node=2, domain={0}))
        sim_pattern.add(Z(node=3, domain={0}))

        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")
        mock_backend = mocker.patch("graphix.visualization_interactive.StatevectorBackend")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.determine_figsize.return_value = (14.0, 7.0)
        mock_place_paths = MagicMock(return_value=({}, {}))
        mock_vis_obj.get_layout.return_value = (
            {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
            mock_place_paths,
            {},
        )

        backend_instance = mock_backend.return_value
        backend_instance.measure.return_value = 1

        viz = InteractiveGraphVisualizer(sim_pattern, enable_simulation=True)

        active, measured, _, corrections, results = viz._update_graph_state(len(sim_pattern))

        assert 0 in measured
        assert 1 in measured
        assert 2 in active
        assert 3 in active
        assert results[0] == 1
        assert results[1] == 1
        assert backend_instance.measure.call_count == 2
        backend_instance.correct_byproduct.assert_called()
        assert "Z" in corrections.get(3, set())

        viz.ax_graph = MagicMock()
        viz.ax_commands = MagicMock()
        viz.slider = MagicMock()
        viz._update(len(sim_pattern))

        mock_vis_obj.draw_node_labels.assert_called()
        annotate_calls = viz.ax_graph.annotate.call_args_list
        label_strings = [call.args[0] for call in annotate_calls if call.args]
        assert any("m=" in str(lbl) for lbl in label_strings)
        assert not any(str(lbl).endswith(("\n=1", "\n=0")) for lbl in label_strings)

    def test_graph_state_without_simulation(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test that topology tracking works and results are empty when simulation is disabled."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.determine_figsize.return_value = (14.0, 7.0)
        mock_place_paths = MagicMock(return_value=({}, {}))
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, mock_place_paths, {})

        viz = InteractiveGraphVisualizer(pattern, enable_simulation=False)

        active, measured, _, _, results = viz._update_graph_state(len(pattern))

        assert 0 in measured
        assert 1 in measured
        assert 2 in active
        assert results == {}

        viz.ax_graph = MagicMock()
        viz.ax_commands = MagicMock()
        viz.slider = MagicMock()
        viz._update(len(pattern))

        assert viz.ax_commands.text.call_count > 0
        mock_vis_obj.draw_node_labels.assert_called()

    def test_visualize(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test the main visualize method (smoke test)."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")
        mock_show = mocker.patch("matplotlib.pyplot.show")
        mocker.patch("graphix.visualization_interactive.Slider")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.determine_figsize.return_value = (14.0, 7.0)
        mock_place_paths = MagicMock(return_value=({}, {}))
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, mock_place_paths, {})

        viz = InteractiveGraphVisualizer(pattern)
        viz.visualize()

        mock_show.assert_called_once()
        assert viz.ax_commands is not None
        assert viz.ax_graph is not None
        assert viz.slider is not None
        assert viz.btn_next is not None
        assert viz.btn_prev is not None

    def test_navigation_and_events(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test step navigation boundaries, keyboard dispatch, and pick-event slider sync."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.determine_figsize.return_value = (14.0, 7.0)
        mock_place_paths = MagicMock(return_value=({}, {}))
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, mock_place_paths, {})

        viz = InteractiveGraphVisualizer(pattern)
        viz.slider = MagicMock()
        viz.total_steps = 10
        viz.current_step = 5

        viz._prev_step(None)
        viz.slider.set_val.assert_called_with(4)
        viz._next_step(None)
        viz.slider.set_val.assert_called_with(6)

        viz.current_step = 0
        viz.slider.reset_mock()
        viz._prev_step(None)
        viz.slider.set_val.assert_not_called()

        viz.current_step = 10
        viz.slider.reset_mock()
        viz._next_step(None)
        viz.slider.set_val.assert_not_called()

        key_event = MagicMock()
        key_event.key = "right"
        mock_next = mocker.patch.object(viz, "_next_step")
        viz._on_key(key_event)
        mock_next.assert_called_once()

        key_event.key = "left"
        mock_prev = mocker.patch.object(viz, "_prev_step")
        viz._on_key(key_event)
        mock_prev.assert_called_once()

        viz.ax_graph = MagicMock()
        viz.ax_commands = MagicMock()
        viz.total_steps = len(pattern)  # restore after manual override
        viz.current_step = 0
        viz._update(len(pattern))
        mock_artist = MagicMock(spec=Text)
        mock_artist.index = 5
        pick_event = MagicMock()
        pick_event.artist = mock_artist
        viz.slider.reset_mock()
        viz._on_pick(pick_event)
        viz.slider.set_val.assert_called_with(6)

    def test_draw_graph_rendering(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test edge and node delegation to GraphVisualizer, and silent exception handling."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.determine_figsize.return_value = (14.0, 7.0)
        mock_place_paths = MagicMock(return_value=({}, {}))
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, mock_place_paths, {})

        viz = InteractiveGraphVisualizer(pattern)
        viz.ax_graph = MagicMock()
        viz.slider = MagicMock()

        viz._update(5)
        mock_vis_obj.draw_edges_with_routing.assert_called()
        assert "edge_colors" in mock_vis_obj.draw_edges_with_routing.call_args.kwargs

        viz._update(len(pattern))
        mock_vis_obj.draw_nodes_role.assert_called()
        assert "node_facecolors" in mock_vis_obj.draw_nodes_role.call_args.kwargs
        assert "node_edgecolors" in mock_vis_obj.draw_nodes_role.call_args.kwargs

        viz.ax_graph.clear.side_effect = ValueError("boom")
        mocker.patch("traceback.print_exc")
        mock_state: tuple[set[int], set[int], list[tuple[int, int]], dict[int, set[str]], dict[int, int]] = (
            set(), set(), [], {}, {}
        )
        viz._draw_graph(mock_state)

    def test_draw_graph_no_flow_and_plane_annotation(self, mocker: MagicMock) -> None:
        """Test the no-flow else branch and plane label annotation for active nodes."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.determine_figsize.return_value = (14.0, 7.0)
        mock_place_paths = MagicMock(return_value=({(0, 1): [(0.0, 0.0), (1.0, 0.0)]}, None))
        mock_vis_obj.get_layout.return_value = (
            {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)},
            mock_place_paths,
            None,
        )
        mock_meas = MagicMock()
        mock_meas.to_plane_or_axis.return_value.name = "XY"
        mock_vis_obj.og.measurements = {0: mock_meas, 1: mock_meas}
        mock_vis_obj.og.input_nodes = []

        no_flow_pattern = Pattern(
            input_nodes=[0, 1],
            cmds=[N(node=0), N(node=1), E(nodes=(0, 1)), M(node=0), M(node=1)],
        )
        viz = InteractiveGraphVisualizer(no_flow_pattern, enable_simulation=False)
        viz.ax_graph = MagicMock()
        viz.ax_commands = MagicMock()
        viz.slider = MagicMock()
        viz.node_positions = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)}

        viz._update(len(no_flow_pattern))
        mock_vis_obj.draw_edges_with_routing.assert_called()
        mock_vis_obj.draw_flow_arrows.assert_not_called()

        viz.current_step = 0
        viz.ax_graph.reset_mock()
        viz._update(1)
        annotate_calls = viz.ax_graph.annotate.call_args_list
        plane_labels = [c.args[0] for c in annotate_calls if c.args and isinstance(c.args[0], str)]
        assert any(lbl == "XY" for lbl in plane_labels)

    def test_draw_command_list_early_return(self, mocker: MagicMock) -> None:
        """Test that _draw_command_list returns without drawing when the window has no visible commands."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.determine_figsize.return_value = (14.0, 7.0)
        mock_vis_obj.get_layout.return_value = ({}, {}, {})

        empty_pattern = Pattern(input_nodes=[])
        viz = InteractiveGraphVisualizer(empty_pattern, enable_simulation=False)
        viz.ax_commands = MagicMock()
        viz.current_step = 0

        viz._draw_command_list({})
        viz.ax_commands.text.assert_not_called()


class TestGraphVisualizerSharedAPI:
    """Tests for the shared drawing API exposed by GraphVisualizer."""

    def test_get_label_fontsize(self) -> None:
        """Test font-size computation for small, large, and custom-base node numbers."""
        assert GraphVisualizer.get_label_fontsize(0) == 12
        assert GraphVisualizer.get_label_fontsize(99) == 12
        large = GraphVisualizer.get_label_fontsize(100)
        assert 7 <= large < 12
        assert GraphVisualizer.get_label_fontsize(0, base_size=10) == 10
        custom = GraphVisualizer.get_label_fontsize(1000, base_size=10)
        assert 7 <= custom < 10

    def test_draw_nodes_role_with_overrides(self) -> None:
        """Test draw_nodes_role applies per-node colour overrides."""
        mock_og = MagicMock()
        mock_og.graph.nodes.return_value = [0, 1, 2]
        mock_og.input_nodes = [0]
        mock_og.output_nodes = [2]
        mock_og.measurements = {0: MagicMock(), 1: MagicMock(), 2: MagicMock()}

        vis = GraphVisualizer(og=mock_og)
        ax = MagicMock()
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}

        vis.draw_nodes_role(
            ax,
            pos,
            node_facecolors={0: "yellow", 1: "pink"},
            node_edgecolors={0: "green"},
        )

        scatter_calls = ax.scatter.call_args_list
        assert len(scatter_calls) == 3
        assert scatter_calls[0].kwargs["facecolors"] == "yellow"
        assert scatter_calls[0].kwargs["edgecolors"] == "green"
        assert scatter_calls[1].kwargs["facecolors"] == "pink"
        assert scatter_calls[1].kwargs["edgecolors"] == "black"
        assert scatter_calls[2].kwargs["facecolors"] == "lightgray"
        assert scatter_calls[2].kwargs["edgecolors"] == "black"

    def test_draw_edges(self) -> None:
        """Test draw_edges with and without an edge subset."""
        mock_og = MagicMock()
        mock_og.graph.edges.return_value = [(0, 1), (1, 2), (2, 3)]

        vis = GraphVisualizer(og=mock_og)
        ax = MagicMock()
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)}

        vis.draw_edges(ax, pos, edge_subset=[(0, 1), (2, 3)])
        assert ax.plot.call_count == 2

        ax.reset_mock()
        vis.draw_edges(ax, pos)
        assert ax.plot.call_count == 3
